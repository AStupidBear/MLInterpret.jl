using MLInterpret
using Test

X = DataFrame(rand(Float32, 1000, 10))
y = X.mean(axis = 1)

@from lightgbm imports LGBMRegressor
model = LGBMRegressor()
model.fit(X, y)

interpret(model, X, y)
interpret(X, y)