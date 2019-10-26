module MLI

using Random, Statistics, Combinatorics, ProgressMeter, Reexport
@reexport using PyCall, PyCallUtils, Pandas

export interpret, dai_interpret, sbrl_interpret, fit_surrogate

hasproperty(pyo, s) = !ispynull(pyo) && PyCall.hasproperty(pyo, s)

bash(cmd) = run(`bash -c "$cmd"`)

macro savefig(dst, ex)
    dst, ex = esc(dst), esc(ex)
    quote
        @imports matplotlib.pyplot as plt
        plt.matplotlib.use("agg")
        plt.close(plt.gcf())
        $ex; mkpath(dirname($dst))
        plt.savefig($dst)
    end
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

istree(model) = occursin(r"gbm|booster|tree|forest"i, string(model))

function interpret(model, X, y; nskater = 50000, nsurro = 500000, nshap = 50000, nsbrl = 20000)
    X.columns = X.columns.astype("str")
    tree = istree(model) ? model : fit_surrogate(X, y)
    pyo =  hasproperty(model, :predict) ? model : tree
    @indir "mli" begin
        skater_interpret(pyo.predict, X, y, nskater)
        surrogate_interpret(pyo.predict, X, y, nsurro)
        cols = shap_interpret(tree, X, y, nshap)
        shap2_interpret(tree, X, y, 3000, cols = cols)
    end
end

interpret(X, y; ka...) = interpret(PyNULL(), X, y; ka...)

function sample(X, y, nsample)
    nsample >= length(y) && return X, y
    X = X.sample(nsample, random_state = 12345)
    y = y.sample(nsample, random_state = 12345)
    X.reset_index(inplace = true, drop = true)
    y.reset_index(inplace = true, drop = true)
    return X, y
end

function fit_surrogate(X, y; max_depth = 8)
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

function sbrl_interpret(X, y, nsample; sample_method = "sample", cols = X.columns)
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

function surrogate_interpret(model, X, y, nsample)
    X, y = sample(X, y, nsample)
    @from skater.model imports InMemoryModel
    @from skater.core.global_interpretation.tree_surrogate imports TreeSurrogate
    skater_model = InMemoryModel(model, feature_names = X.columns, examples = X.iloc[1:2])
    for n in broadcast(^, 2, [2, 3, 4])
        surrogate_explainer = TreeSurrogate(skater_model, max_leaf_nodes = 2n)
        surrogate_explainer.fit(X, y, use_oracle = false)
        surrogate_explainer.plot_global_decisions(file_name = "surrogate_tree-$n.png")
    end
end

function skater_interpret(model, X, y, nsample; ntop = 20)
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
        @savefig "pdp/$n.pdf" ppd([c], skater_model, with_variance = true, grid_resolution = 10, sample = false)
    end
    bash("pdfcat -o pdp.pdf pdp/*.pdf && rm -rf pdp")
end

function dai_interpret(X, y, nsample)
    start_dai()
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

function shap_interpret(model, X, y, nsample; ntop = 20)
    X, y = sample(X, y, nsample)
    @from shap imports TreeExplainer, summary_plot, dependence_plot, force_plot, save_html
    explainer = TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    cols = shap_topfeas(shap_vals, X, ntop)
    @savefig "summary/dot.pdf" summary_plot(shap_vals, X)
    @savefig "summary/bar.pdf" summary_plot(shap_vals, X, plot_type = "bar")
    for (n, c) in enumerate(cols)
        @savefig "depend/$n.pdf" dependence_plot(c, shap_vals, X)
    end
    bash("pdfcat -o shap.pdf summary/*.pdf depend/*.pdf && rm -rf summary depend")
    i = randsubseq(1:length(X), min(1, 5000 / length(X)))
    j = [findfirst(c .== X.columns) for c in cols]
    html = force_plot(explainer.expected_value, shap_vals[i, j], X.iloc[i, j])
    save_html("force_plot.html", html)
    return cols
end

function shap2_interpret(model, X, y, nsample; ntop = 7, cols = X.columns)
    @from copy imports deepcopy
    X, model = X[cols], deepcopy(model)
    model.fit(X, y, eval_set = [(X, y)])
    X, y = sample(X, y, nsample)
    @from shap imports TreeExplainer, summary_plot, dependence_plot
    explainer = TreeExplainer(model)
    shap_vals = explainer.shap_interaction_values(X)
    @savefig "summary/dot.pdf" summary_plot(shap_vals, X)
    @savefig "summary/bar.pdf" summary_plot_bar2(shap_vals, X)
    for (n, cc) in enumerate(combinations(cols[1:ntop], 2))
        @savefig "depend/$n.pdf" dependence_plot(cc, shap_vals, X)
    end
    bash("pdfcat -o shap2.pdf summary/*.pdf depend/*.pdf && rm -rf summary depend")
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
    plt.yticks(axes(M, 1) .- 1, cols, rotation = 60)
    plt.xticks(axes(M, 2) .- 1, cols, rotation = 60)
    plt.gca().xaxis.tick_top()
end

install_dai() = run(`bash $(joinpath(@__DIR__, "../deps/dai-install.sh"))`)

start_dai() = run(`bash $(joinpath(@__DIR__, "../deps/dai-start.sh"))`)

end # module