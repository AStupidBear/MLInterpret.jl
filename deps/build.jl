using Pkg, BinDeps, PyCall
using PyCall: python, conda, Conda
using BinDeps: generate_steps, getallproviders, lower, PackageManager

!Sys.islinux() && exit()

if conda && Conda.version("python") >= v"3.7"
    # skater does not support python3.7
    Conda.add("python=3.6")
    Pkg.build("PyCall")
end

if isnothing(Sys.which("sudo")) # in docker
    try run(`apt update`) catch end
    try run(`yum update`) catch end
end

@BinDeps.setup

gcc = library_dependency("gcc")
dot = library_dependency("dot")
python3 = library_dependency("python3")

provides(AptGet, Dict("g++" => gcc, "python3-dev" => python3, "graphviz" => dot))
provides(Yum, Dict("gcc-c++" => gcc, "python3-devel" => python3, "graphviz" => dot))

for dep in bindeps_context.deps
    dp, opts = getallproviders(dep, PackageManager)[1]
    cmd = lower(generate_steps(dep, dp, opts)).steps[1]
    i = findfirst(x -> x == "install", cmd.exec)
    insert!(cmd.exec, i + 1, "-y")
    run(cmd)
end

run(`$python -m pip install --user pandas sklearn matplotlib lightgbm ipython shap keras tzlocal PyPDF2 unidecode skater pdpbox`)