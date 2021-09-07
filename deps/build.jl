using Pkg, BinDeps, PyCall
using PyCall: python, conda, Conda
using BinDeps: generate_steps, getallproviders, lower, PackageManager

!Sys.islinux() && exit()

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
    println(cmd)
    try run(cmd) catch end
end

run(`$python -m pip install --user pandas "scikit-learn<=0.22.2" matplotlib lightgbm ipython shap keras tzlocal PyPDF2 unidecode pdpbox`)
run(`$python -m pip install git+https://github.com/oracle/Skater.git`)

buildsh = joinpath(@__DIR__, "build.sh")
ENV["JULIA_DEPOT_PATH"] = DEPOT_PATH[1]
run(`bash $buildsh`)
