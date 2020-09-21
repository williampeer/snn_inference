from Dev.brian2_custom_network_opt import *
import nevergrad as ng
from Log import Logger

# TODO: PSO, NGO, CMA(?)
for optim in [ng.optimizers.DE, ng.optimizers.CMA, ng.optimizers.PSO, ng.optimizers.NGO]:
    N = 12
    w_mean = 0.3; w_var = 0.5; rand_ws = (w_mean - w_var) + 2 * w_var * np.random.random((N ** 2))
    instrum = ng.p.Instrumentation(w=ng.p.Array(init=rand_ws).set_bounds(-1., 1.),
                                   f_I=ng.p.Array(init=0.4 * np.ones((N,))).set_bounds(0.01, 0.99),
                                   C_m=ng.p.Array(init=1.5 * np.ones((N,))).set_bounds(1., 3.),
                                   G=ng.p.Array(init=0.8 * np.ones((N,))).set_bounds(0.01, 0.99),
                                   R_I=ng.p.Array(init=18. * np.ones((N,))).set_bounds(10., 30.),
                                   f_v=ng.p.Array(init=0.14 * np.ones((N,))).set_bounds(0.01, 0.99),
                                   E_L=ng.p.Array(init=-65. * np.ones((N,))).set_bounds(-90., -30.),
                                   loss_fn=ng.p.Choice(['van_rossum_dist', 'poisson_nll', 'gamma_factor']))

    optimizer = optim(parametrization=instrum, budget=200)

    logger = Logger(log_fname='brian2_network_nevergrad_optimization_{}')
    logger.log('setup experiment with the optimizer {}'.format(optimizer.__str__()))

    recommendation = optimizer.minimize(run_simulation_for, verbosity=2)

    logger.log('recommendation.value: {}'.format(recommendation.value))

    # hip.Experiment.from_iterable(enumerate(recommendation.value)).display()
