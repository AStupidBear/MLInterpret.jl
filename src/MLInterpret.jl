module MLInterpret

using Random, Statistics, Combinatorics, Distributed
using ProgressMeter, Reexport
@reexport using PyCall, PyCallUtils, PandasLite
using PyCall: python
using PandasLite: PandasWrapped
import StatsBase: sample, Weights
using Pkg.GitTools: clone

export interpret, fit_surrogate, dai_interpret, sbrl_interpret, dnn_interpret

hasproperty(pyo, s) = !ispynull(pyo) && PyCall.hasproperty(pyo, s)

bash(cmd) = run(`bash -c "$cmd"`)

const pdfcat = joinpath(@__DIR__, "pdfcat")

macro savefig(dst, ex)
    dst, ex = esc(dst), esc(ex)
    quote
        @imports matplotlib.pyplot as plt
        plt.matplotlib.use("agg")
        plt.close(plt.gcf())
        $ex; mkpath(dirname($dst))
        plt.savefig($dst, bbox_inches = "tight")
    end
end

macro savefig(dir, n, ex)
    quote
        dst = joinpath($dir, lpad($n, 2, "0") * ".pdf")
        @savefig dst $ex
    end |> esc
end

macro indir(dir, ex)
    cwd = gensym()
    quote
        mkpath($dir); $cwd = pwd(); cd($dir)
        try $ex finally cd($cwd) end
    end |> esc
end

macro redirect(src, ex)
    src = src == :devnull ? "/dev/null" : src
    quote
        io = open($(esc(src)), "a")
        o, e = stdout, stderr
        redirect_stdout(io)
        redirect_stderr(io)
        try
            $(esc(ex)); sleep(0.01)
        finally
            flush(io); close(io)
            redirect_stdout(o)
            redirect_stderr(e)
        end
    end
end

istree(model) = occursin(r"gbm|booster|tree|forest"i, string(model)) && hasproperty(model, :predict)

function interpret(model, X, y; nskater = 50000, nsurro = 500000, nshap = 50000, nshap2 = 3000)
    X.columns = X.columns.astype("str")
    X = X.fillna(X.mean(axis = 0).fillna(0))
    tree = istree(model) ? model : fit_surrogate(X, y)
    pyo =  hasproperty(model, :predict) ? model : tree
    @indir "mli" begin
        surrogate_interpret(pyo.predict, X, y, nsample = nsurro)
        cols = shap_interpret(tree, X, nsample = nshap)
        pdp_interpret(pyo, X, nsample = nskater, cols = cols)
        shap2_interpret(X, y, nsample = nshap2, cols = cols)
        skater_interpret(pyo.predict, X, y, nsample = nskater)
    end
end

interpret(X, y; ka...) = interpret(PyNULL(), X, y; ka...)

function sample(x::PandasWrapped, nsample)
    nsample >= length(x) && return x
    x = x.sample(nsample, random_state = 12345)
    x.reset_index(inplace = true, drop = true)
    return x
end

sample(X::PandasWrapped, y, nsample) = sample(X, nsample), sample(y, nsample)

function fit_surrogate(X, y; max_depth = 8)
    @from unidecode imports unidecode
    cols = [filter(!isspace, unidecode(c)) for c in X.columns]
    X = DataFrame(X, columns = cols)
    # TODO: tuning parameters
    @from lightgbm imports LGBMRegressor
    @from sklearn.metrics imports r2_score
    surrogate = LGBMRegressor(
        n_jobs = 20, max_bin = 15, boosting_type = "rf",
        subsample = 0.632, colsample_bytree = 0.632,
        subsample_freq = 1, num_leaves = 2^max_depth - 1
    )
    surrogate.fit(X, y, eval_set = [(X, y)])
    ŷ = surrogate.predict(X)
    println("r2: ", r2_score(y, ŷ))
    return surrogate
end

function sbrl_interpret(X, y; nsample = length(y), sample_method = "sample", cols = X.columns)
    @from skater.core.global_interpretation.interpretable_models.brlc imports BRLC
    @from skater.core.global_interpretation.interpretable_models.bigdatabrlc imports BigDataBRLC
    if sample_method == "sample"
        sbrl_model = BRLC(iterations = 30000, n_chains = 10, max_rule_len = 4, drop_features = true)
        X, y = sample(X[cols], y, nsample)
    elseif sample_method == "surrogate"
        pct = min(nsample / length(y), 1)
        sbrl_model = BigDataBRLC(iterations = 30000, n_chains = 10, drop_features = true, sub_sample_percentage = pct)
    end
    @indir "mli" begin
        sbrl_model.fit(X[cols], Float32.(y .> quantile(y, 0.7)))
        @redirect "sbrl.txt" sbrl_model.print_model()
        # sbrl_model.save_model("sbrl.pkl")
        bash("rm tdata_*")
    end
end

function surrogate_interpret(model, X, y; nsample = length(X))
    X, y = sample(X, y, nsample)
    @from skater.model imports InMemoryModel
    @from skater.core.global_interpretation.tree_surrogate imports TreeSurrogate
    skater_model = InMemoryModel(model, feature_names = X.columns, examples = X.iloc[1:2])
    for n in broadcast(^, 2, [3, 4, 5])
        surrogate_explainer = TreeSurrogate(skater_model, max_leaf_nodes = 2n)
        surrogate_explainer.fit(X, y, use_oracle = false)
        surrogate_explainer.plot_global_decisions(file_name = "surrogate_tree-$n.png")
    end
end

function skater_interpret(model, X, y; nsample = length(X), ntop = 20)
    X, y = sample(X, y, nsample)
    @from skater.core.explanations imports Interpretation
    @from skater.model imports InMemoryModel
    interpreter = Interpretation(X, y)
    skater_model = InMemoryModel(model, feature_names = X.columns, examples = X.iloc[1:2])
    sr = interpreter.feature_importance.feature_importance(skater_model, n_samples = 20000)
    sr.sort_values(ascending = false, inplace = true)
    sr.to_csv("perturb_feaimpt.csv", header = false, encoding = "gbk")
    ppd = interpreter.partial_dependence.plot_partial_dependence
    cols = sr.head(ntop).index.tolist()
    @showprogress for (n, c) in enumerate(cols)
        @savefig "pdp" n ppd([c], skater_model, with_variance = true, grid_resolution = 10, sample = false)
    end
    bash("$python $pdfcat -o pdp.pdf pdp/*.pdf && rm -rf pdp")
end

function pdp_interpret(model, X; nsample = length(X), ntop = 7, cols = X.columns)
    X = sample(X, nsample)
    @from pdpbox.info_plots imports actual_plot, actual_plot_interact
    @from warnings imports filterwarnings
    filterwarnings("ignore"; :module => "pdpbox")
    plot_params = Dict("font_family" => "sans-serif")
    for (n, c) in enumerate(cols)
        @savefig "actual" n actual_plot(model, X, c, c, predict_kwds = Dict(), plot_params = plot_params)
    end
    bash("$python $pdfcat -o actual.pdf actual/*.pdf && rm -rf actual")
    for (n, cc) in enumerate(combinations(cols[1:min(ntop, end)], 2))
        @savefig "actual2" n actual_plot_interact(model, X, cc, cc, predict_kwds = Dict(), plot_params = plot_params)
    end
    bash("$python $pdfcat -o actual2.pdf actual2/*.pdf && rm -rf actual2")
end

function dai_interpret(X, y; nsample = length(X))
    start_dai()
    whl="http://localhost:12345/static/h2oai_client-1.7.0-py3-none-any.whl"
    run(pipeline(`$python -m pip install $whl`, stdout = devnull))
    X, y = sample(X, y, nsample)
    target = Series(y, name = "target")
    pred = Series(y, name = "pred")
    df = pd.concat([X, target, pred], axis = 1)
    df.to_parquet("pred.parquet")
    @from h2oai_client imports Client
    h2oai = Client(address = "http://127.0.0.1:12345", username = "abcd", password = "dcba")
    pred = h2oai.upload_dataset_sync("pred.parquet")
    mli = h2oai.run_interpretation_sync("", pred.key, "target", prediction_col = "pred")
    touch(mli.description * ".mli")
end

function shap_interpret(model, X; nsample = length(X), ntop = 20)
    X = sample(X, nsample)
    @from shap imports TreeExplainer, summary_plot, dependence_plot, force_plot, save_html
    explainer = TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    cols = shap_topfeas(shap_vals, X, ntop)
    @savefig "summary/dot.pdf" summary_plot(shap_vals, X)
    @savefig "summary/bar.pdf" summary_plot(shap_vals, X, plot_type = "bar")
    for (n, c) in enumerate(cols)
        @savefig "depend" n dependence_plot(c, shap_vals, X)
    end
    bash("$python $pdfcat -o shap.pdf summary/*.pdf depend/*.pdf && rm -rf summary depend")
    i = randsubseq(1:length(X), min(1, 5000 / length(X)))
    j = [findfirst(c .== X.columns) for c in cols]
    html = force_plot(explainer.expected_value, shap_vals[i, j], X.iloc[i, j])
    save_html("force_plot.html", html)
    return cols
end

function shap2_interpret(X, y; nsample = length(X), ntop = 7, cols = X.columns)
    model = fit_surrogate(X[cols], y)
    X, y = sample(X[cols], y, nsample)
    @from shap imports TreeExplainer, summary_plot, dependence_plot
    explainer = TreeExplainer(model)
    shap_vals = explainer.shap_interaction_values(X)
    @savefig "summary/dot.pdf" summary_plot(shap_vals, X)
    @savefig "summary/bar.pdf" summary_plot_bar2(shap_vals, X)
    for (n, cc) in enumerate(combinations(cols[1:min(ntop, end)], 2))
        @savefig "depend" n dependence_plot(cc, shap_vals, X)
    end
    bash("$python $pdfcat -o shap2.pdf summary/*.pdf depend/*.pdf && rm -rf summary depend")
end

function shap_topfeas(shap_vals, X, ntop = size(X, 2))
    feaimpt = vec(sum(abs, shap_vals, dims = 1))
    is = sortperm(feaimpt, rev = true)[1:min(ntop, end)]
    collect(X.columns[is])
end

function summary_plot_bar2(shap_vals, X)
    M = dropdims(sum(abs, shap_vals, dims = 1), dims = 1)
    for i in axes(M, 1) M[i, i] = 0 end
    feaimpt = vec(sum(M, dims = 2))
    is = sortperm(feaimpt, rev = true)
    M, cols = M[is, is], X.columns[is]
    @imports matplotlib.pyplot as plt
    plt.imshow(M)
    plt.yticks(axes(M, 1) .- 1, cols)
    plt.xticks(axes(M, 2) .- 1, cols)
    plt.gca().xaxis.tick_top()
end

function dnn_interpret(a...; ka...)
    id, = addprocs(1)
    @eval @everywhere using MLI
    remotecall_fetch(_dnn_interpret, id, a...; ka...)
    rmprocs(id)
end

function _dnn_interpret(h5::String, a...; ka...)
    @from keras.models imports load_model
    model = load_model(h5, compile = false)
    _dnn_interpret(model, a...; ka...)
end

function _dnn_interpret(model, X; alg = "deeplift", width = 20, nsample = 5000, nviz = 100, recep = 0, cols = [])
    X = sample_clip(X, nsample, recep)
    if model.output.shape.ndims == 3
        @from keras.layers imports Lambda
        @from keras.models imports Model
        output = Lambda(py"lambda x: x[:, -1, :]")(model.output)
        model = Model(model.inputs, output)
    end
    Xviz = sample_rank(X, vec(model.predict(X)), nviz)
    @from shap imports DeepExplainer, GradientExplainer, image_plot
    class = alg == "deeplift" ? DeepExplainer : GradientExplainer
    shap_vals, = class(model, X).shap_values(Xviz)
    @imports matplotlib.pyplot as pl
    @from shap.plots imports colors
    @from matplotlib.backends.backend_pdf imports PdfPages
    @imports numpy as np
    max_val = np.nanpercentile(abs.(shap_vals), 99.9)
    xmax = np.nanpercentile(X, 99.9)
    xmin = np.nanpercentile(X, 0.1)
    smax = np.nanpercentile(abs.(shap_vals), 99.9)
    pdf = PdfPages("dnn.pdf")
    @showprogress for n in 1:size(Xviz, 1)
        x = permutedims(Xviz[n, :, :])
        sv = permutedims(shap_vals[n, :, :])
        figsize = [3 * (size(x, 1) + 1), 2.5 * (size(x, 2) + 1)]
        if figsize[1] > width
            @. figsize = width * figsize / figsize[1]
        end
        fig, axes = pl.subplots(ncols = 2, sharey = true, figsize = figsize)
        axes[1].imshow(x, cmap = pl.get_cmap("gray"), vmin = xmin, vmax = xmax)
        if !isempty(cols)
            axes[1].set_yticks(0:size(x, 1))
            axes[1].set_yticklabels(cols)
            labelsize = 5 * figsize[1] / size(x, 1)
            axes[1].tick_params(axis = "y", labelsize = labelsize, length = 0)
            axes[1].get_xaxis().set_visible(false)
        else
            axes[1].set_axis_off()
        end
        axes[2].imshow(x, cmap = pl.get_cmap("gray"), alpha = 0.15)
        axes[2].imshow(sv, cmap = colors.red_transparent_blue, vmin = -smax, vmax = smax)
        axes[2].set_axis_off()
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches = "tight")
        pl.close(fig)
    end
    pdf.close()
end

function sample_clip(X, nsample, recep)
    N, T, F = size(X)
    recep = recep == 0 ? T : recep
    nsample = min(nsample, N * T ÷ recep)
    Xs = zeros(Float32, nsample, recep, F)
    for ns in 1:nsample
        n, t = rand(1:N), rand(1:(T - recep + 1))
        Xs[ns, :, :] = X[n, t:(t + recep - 1), :]
    end
    return Xs
end

function sample_rank(X, y, nsample)
    nsample = min(nsample, length(y))
    perm = sortperm(y, by = abs, rev = true)
    wv = Weights(1 ./ axes(y, 1)[invperm(perm)])
    ns = sample(1:length(y), wv, nsample, replace = false)
    return X[ns, :, :]
end

const deps = joinpath(@__DIR__, "../deps")

install_dai() = run(`bash $deps/dai-install.sh`)

start_dai() = run(`bash $deps/dai-start.sh`)

function install_brl()
    skater = mktempdir()
    clone("https://github.com/oracle/Skater.git", skater)
    run(`sed -i "s|sudo python|$python|g" $skater/setup.sh`)
    run(`sed -i "s|.*install python.*||g" $skater/setup.sh`)
    run(`sed -i "s|apt-get install|apt-get install -y|g" $skater/setup.sh`)
    run(`bash -c "cd $skater && $python setup.py install --ostype=linux-ubuntu --rl=True"`)
end

end # module
