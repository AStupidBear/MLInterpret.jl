using PyCall: python, Conda
using Pkg.GitTools: clone

isnothing(Sys.which("dot")) && run(`sudo apt-get install -y graphviz`)

run(`$python -m pip install pandas sklearn matplotlib lightgbm shap keras tzlocal PyPDF2`)

Conda.add("gxx_linux-64")
skater = mktempdir()
clone("https://github.com/oracle/Skater.git", skater)
run(`sed -i "s|sudo python|$python|g" $skater/setup.sh`)
run(`sed -i "s|.*install python.*||g" $skater/setup.sh`)
run(`sed -i "s|apt-get install|apt-get install -y|g" $skater/setup.sh`)
run(`bash -c "cd $skater && $python setup.py install --ostype=linux-ubuntu --rl=True"`)