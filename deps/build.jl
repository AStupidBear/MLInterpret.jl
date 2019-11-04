using BinDeps, PyCall
using PyCall: python
using BinDeps: generate_steps, getallproviders, lower, PackageManager

@BinDeps.setup
gcc = library_dependency("gcc")
dot = library_dependency("dot")
if Sys.islinux()
    provides(AptGet, Dict("gcc" => gcc, "graphviz" => dot))
    provides(Yum, Dict("gcc-c++" => gcc, "graphviz" => dot))
end
for dep in bindeps_context.deps
    dp, opts = getallproviders(dep, PackageManager)[1]
    run(lower(generate_steps(dep, dp, opts)))
end

run(`$python -m pip install --user pandas sklearn matplotlib lightgbm shap keras tzlocal PyPDF2 unidecode skater`)