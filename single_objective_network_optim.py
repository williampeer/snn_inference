import nevergrad as ng

import IO
from Dev.brian2_custom_network_opt import *
from Log import Logger
from experiments import zip_dicts
from plot import plot_all_param_pairs_with_variance, plot_spiketrains_side_by_side

num_exps = 20; budget = 1000
# num_exps = 3; budget = 40


params_by_optim = {}
optim_names = ['DE', 'CMA', 'PSO', 'NGO']; optim_ctr = 0
for optim in [ng.optimizers.DE, ng.optimizers.CMA, ng.optimizers.PSO, ng.optimizers.NGO]:
    cur_optim_name = optim_names[optim_ctr]
    optim_ctr += 1
    # for loss_fn in ['van_rossum_dist', 'poisson_nll', 'gamma_factor']:
    for loss_fn in ['vrdfrd']:
        UUID = IO.dt_descriptor()
        current_plottable_params_for_optim = {}
        other_params_for_optim = {}
        exp_min_losses = []
        for exp_i in range(num_exps):
            N = 12
            w_mean = 0.3; w_var = 0.5; rand_ws = (w_mean - w_var) + 2 * w_var * np.random.random((N ** 2))
            instrum = ng.p.Instrumentation(rate=ng.p.Scalar(init=60.).set_bounds(1., 80.),
                                           w=ng.p.Array(init=rand_ws).set_bounds(-1., 1.),
                                           E_L=ng.p.Array(init=-65. * np.ones((N,))).set_bounds(-90., -30.),
                                           C_m=ng.p.Array(init=1.5 * np.ones((N,))).set_bounds(1., 3.),
                                           G=ng.p.Array(init=0.8 * np.ones((N,))).set_bounds(0.01, 0.99),
                                           R_I=ng.p.Array(init=18. * np.ones((N,))).set_bounds(12., 30.),
                                           f_v=ng.p.Array(init=0.14 * np.ones((N,))).set_bounds(0.01, 0.99),
                                           f_I=ng.p.Array(init=0.4 * np.ones((N,))).set_bounds(0.01, 0.99),

                                           delta_theta_s=ng.p.Array(init=10. * np.ones((N,))).set_bounds(6., 30.),
                                           b_s=ng.p.Array(init=0.3 * np.ones((N,))).set_bounds(0.01, 0.9),
                                           a_v=ng.p.Array(init=0.5 * np.ones((N,))).set_bounds(0.01, 0.9),
                                           b_v=ng.p.Array(init=0.5 * np.ones((N,))).set_bounds(0.01, 0.9),
                                           theta_inf=ng.p.Array(init=-15. * np.ones((N,))).set_bounds(-25., 0.),
                                           delta_V=ng.p.Array(init=6. * np.ones((N,))).set_bounds(0.01, 35.),
                                           I_A=ng.p.Array(init=2. * np.ones((N,))).set_bounds(0.5, 4.),
                                           loss_fn=loss_fn)

            optimizer = optim(parametrization=instrum, budget=budget)

            logger = Logger(log_fname='brian2_network_nevergrad_optimization_budget_{}'.format(budget))
            logger.log('setup experiment with the optimizer {}'.format(optimizer.__str__()))

            recommendation = optimizer.minimize(run_simulation_for, verbosity=2)

            logger.log('recommendation.value: {}'.format(recommendation.value))
            fitted_params = recommendation.value[1]
            cur_plot_params = {}
            exp_min_losses.append(recommendation.loss)
            index_ctr = 0
            for p_i, key in enumerate(fitted_params):
                if key not in ['loss_fn', 'rate', 'w']:
                    if exp_i == 0:
                        current_plottable_params_for_optim[index_ctr] = [np.copy(fitted_params[key])]
                    else:
                        current_plottable_params_for_optim[index_ctr].append(np.copy(fitted_params[key]))
                    cur_plot_params[key] = np.copy(fitted_params[key])
                    index_ctr += 1
                else:
                    if exp_i == 0:
                        other_params_for_optim[key] = [np.copy(fitted_params[key])]
                    else:
                        other_params_for_optim[key].append(np.copy(fitted_params[key]))

            model_spike_train = get_spike_train_for(fitted_params['rate'], other_params_for_optim['w'][exp_i], cur_plot_params.copy())

            _, targets = data_util.get_spike_train_matrix(
                index_last_step=int(0.6 * np.random.rand() * spike_times.shape[0]),
                advance_by_t_steps=time_interval, spike_times=spike_times,
                spike_indices=spike_indices, node_numbers=spike_node_indices)
            plot_spiketrains_side_by_side(model_spike_train, targets, exp_type='single_objective_optim', uuid=UUID,
                                          title='Spike trains model and target ({}, loss: {})'.format(cur_optim_name, recommendation.loss),
                                          fname='spike_trains_optim_{}_exp_num_{}'.format(cur_optim_name, exp_i))



        params_by_optim[cur_optim_name] = zip_dicts(current_plottable_params_for_optim, other_params_for_optim)

        plot_all_param_pairs_with_variance(current_plottable_params_for_optim,
                                           exp_type='single_objective_optim',
                                           uuid=UUID,
                                           target_params=target_parameters,
                                           param_names=list(fitted_params.keys())[2:],
                                           custom_title="KDE projection of 2D model parameter".format(cur_optim_name),
                                           logger=logger, fname='single_objective_KDE_optim_{}'.format(cur_optim_name))

        torch.save(params_by_optim, './saved/single_objective_optim/fitted_params_optim_{}_loss_fn_{}_budget_{}.pt'.format(optim, loss_fn, budget))
        torch.save(exp_min_losses, './saved/single_objective_optim/min_losses_optim_{}_loss_fn_{}_budget_{}.pt'.format(optim, loss_fn, budget))
