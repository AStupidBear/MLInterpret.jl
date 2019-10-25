# Machine Learning Interpretation

## Installation

```julia
julia>]
pkg> add https://github.com/AStupidBear/Pandas.jl.git
pkg> add https://github.com/AStupidBear/PyCallUtils.jl.git
pkg> add https://github.com/AStupidBear/MLI.jl.git
```

## Usage

```julia
using MLI
```

```
X = DataFrame(rand(Float32, 1000, 10))
y = X.mean(axis = 1)
```

```
interpret(X, y)
```

```
@from lightgbm imports LGBMRegressor
model = LGBMRegressor()
model.fit(X, y)
interpret(model, X, y)
```

```
dai_interpret(X, y)
```

```
sbrl_interpret(X, y)
```