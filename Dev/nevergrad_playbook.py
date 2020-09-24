import nevergrad as ng


def square(x, y=1):
    return sum(x + y)


# v1 = ng.p.Scalar(init=1.0).set_bounds(-10., 10.)
# v2 = ng.p.Scalar(init=2.0).set_bounds(-10., 10.)
# instrum = ng.p.Instrumentation(x=ng.p.Array(shape=(1,)), y=ng.p.Scalar())
instrum = ng.p.Instrumentation(x=ng.p.Array(shape=(1,)).set_bounds(0., 100.), y=ng.p.Scalar().set_bounds(-100., 0.))

optimizer = ng.optimizers.DE(parametrization=instrum, budget=500)
recommendation = optimizer.minimize(square)  # best value
print('recommendation.value:', recommendation.value)
# >>> [0.49971112 0.5002944]
