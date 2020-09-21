from nevergrad.functions import MultiobjectiveFunction

from Dev.brian2_custom_network_opt import *
import nevergrad as ng

# TODO: PSO, NGO, CMA(?)
for optim in [ng.optimizers.DE, ng.optimizers.CMA, ng.optimizers.PSO, ng.optimizers.NGO]:
    N = 12
    w_mean = 0.3; w_var = 0.5; rand_ws = (w_mean - w_var) + 2 * w_var * np.random.random((N ** 2))
    instrum = ng.p.Instrumentation(w=ng.p.Array(init=rand_ws).set_bounds(-1., 1.),
                                   f_I=ng.p.Array(init=0.4 * np.ones((N,))).set_bounds(0.01, 0.99),
                                   C_m=ng.p.Array(init=1.5 * np.ones((N,))).set_bounds(1., 3.),
                                   G=ng.p.Array(init=0.8 * np.ones((N,))).set_bounds(0.01, 0.99),
                                   R_I=ng.p.Array(init=18. * np.ones((N,))).set_bounds(15., 30.),
                                   f_v=ng.p.Array(init=0.14 * np.ones((N,))).set_bounds(0.01, 0.99),
                                   E_L=ng.p.Array(init=-65. * np.ones((N,))).set_bounds(-90., -30.),

                                   b_s=ng.p.Array(init=0.3 * np.ones((N,))).set_bounds(0.01, 0.9),
                                   b_v=ng.p.Array(init=0.5 * np.ones((N,))).set_bounds(0.01, 0.9),
                                   a_v=ng.p.Array(init=0.5 * np.ones((N,))).set_bounds(0.01, 0.9),
                                   delta_theta_s=ng.p.Array(init=30. * np.ones((N,))).set_bounds(1., 40.),
                                   delta_V=ng.p.Array(init=12. * np.ones((N,))).set_bounds(0.01, 35.),
                                   theta_inf=ng.p.Array(init=-20. * np.ones((N,))).set_bounds(-25., 0.),
                                   I_A=ng.p.Array(init=8. * np.ones((N,))).set_bounds(0.5, 10.)
                                   )
    budget = 200
    optimizer = optim(parametrization=instrum, budget=budget)

    logger = Logger(log_fname='brian2_network_nevergrad_multiobjective_optimization_budget_{}'.format(budget))
    logger.log('setup experiment with the optimizer {}'.format(optimizer.__str__()))

    multiobjective_fn = MultiobjectiveFunction(multiobjective_function=run_simulation_multiobjective)
    recommendation = optimizer.minimize(multiobjective_fn, verbosity=2)

    logger.log('recommendation.value: {}'.format(recommendation.value))

    logger.log("Random subset:", multiobjective_fn.pareto_front(2, subset="random"))
    logger.log("Loss-covering subset:", multiobjective_fn.pareto_front(2, subset="loss-covering"))
    logger.log("Domain-covering subset:", multiobjective_fn.pareto_front(2, subset="domain-covering"))

    # hip.Experiment.from_iterable(enumerate(recommendation.value)).display()
