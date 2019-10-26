using PyCall: python, Conda
using Pkg.GitTools: clone

if isempty(Sys.which("sudo"))
    run(`apt-get update`)
    run(`apt-get install sudo`)
end

if isempty(Sys.which("gcc"))
    run(`sudo apt-get install -y build-essential`)
end

run(`$python -m pip install pandas sklearn matplotlib lightgbm shap keras tzlocal`)

if isempty(Sys.which("add-apt-repository"))
    run(`sudo apt-get install -y software-properties-common`)
end
Conda.add("gxx_linux-64")
skater = mktempdir()
clone("https://github.com/oracle/Skater.git", skater)
run(`sed -i "s|sudo python|$python|g" $skater/setup.sh`)
run(`sed -i "s|.*install python.*||g" $skater/setup.sh`)
run(`$python $skater/setup.py install --ostype=linux-ubuntu --rl=True`)