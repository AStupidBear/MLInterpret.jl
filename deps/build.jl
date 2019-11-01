using PyCall: python

isnothing(Sys.which("dot")) && run(`sudo apt-get install -y graphviz`)

run(`$python -m pip install pandas sklearn matplotlib lightgbm shap keras tzlocal PyPDF2 unidecode skater`)