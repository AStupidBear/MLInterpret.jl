# Machine Learning Interpretation

## Installation

```julia
using Pkg
pkg"add https://github.com/AStupidBear/Pandas.jl.git"
pkg"add https://github.com/AStupidBear/PyCallUtils.jl.git"
pkg"add https://github.com/AStupidBear/MLI.jl.git"
```

## Usage

```julia
using MLI
```

```julia
X = DataFrame(rand(Float32, 1000, 10))
y = X.mean(axis = 1)
```

```julia
interpret(X, y)
```

```julia
@from lightgbm imports LGBMRegressor
model = LGBMRegressor()
model.fit(X, y)
interpret(model, X, y)
```

```julia
dai_interpret(X, y)
```

```julia
sbrl_interpret(X, y)
```

```bash
curl https://raw.githubusercontent.com/AStupidBear/MLI.jl/master/Dockerfile | docker build -t mli -
docker run -it --init --rm mli
```
