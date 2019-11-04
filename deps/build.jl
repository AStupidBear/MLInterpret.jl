using PyCall: python
using BinDeps

@BinDeps.setup
gcc = library_dependency("gcc", runtime = false)
graphviz = library_dependency("graphviz", runtime = false)
if Sys.islinux()
    provides(AptGet, Dict("gcc" => gcc, "graphviz" => graphviz))
    provides(Yum, Dict("gcc-c++" => gcc, "graphviz" => graphviz))
end
@BinDeps.install

run(`$python -m pip install --user pandas sklearn matplotlib lightgbm shap keras tzlocal PyPDF2 unidecode skater`)