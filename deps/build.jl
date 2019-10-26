using PyCall: python, Conda
using Pkg.GitTools: clone

install(bin, pkg) = isnothing(Sys.which(bin)) && run(`sudo apt-get install -y $pkg`)

try run(`apt-get update`) catch end
install("sudo", "sudo")
install("wget", "wget")
install("gcc", "build-essential")
install("add-apt-repository", "software-properties-common")

run(`$python -m pip install pandas sklearn matplotlib lightgbm shap keras tzlocal`)

Conda.add("graphviz")
Conda.add("gxx_linux-64")

skater = mktempdir()
clone("https://github.com/oracle/Skater.git", skater)
run(`sed -i "s|sudo python|$python|g" $skater/setup.sh`)
run(`sed -i "s|.*install python.*||g" $skater/setup.sh`)
run(`bash -c "cd $skater && $python setup.py install --ostype=linux-ubuntu --rl=True"`)