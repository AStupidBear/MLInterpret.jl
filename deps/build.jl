using PyCall: python
run(`$python -m pip install pandas sklearn matplotlib lightgbm shap`)
skater_install = joinpath(@__DIR__, "skater-install.sh")
dai_install = joinpath(@__DIR__, "dai-install.sh")
run(`bash $skater_install`)
run(`bash $dai_install`)
