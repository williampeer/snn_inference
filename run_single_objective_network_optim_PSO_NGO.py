import nevergrad as ng

import IO
from Dev.brian2_custom_network_opt import *
from Log import Logger
from experiments import zip_dicts
from plot import plot_all_param_pairs_with_variance, plot_spiketrains_side_by_side


def main(argv):
    print('Argument List:', str(argv))

    output_fnames_rate_0_6 = ['generated_spike_train_random_glif_1_model_t_300s_rate_0_6.mat',
                              'generated_spike_train_random_glif_2_model_t_300s_rate_0_6.mat',
                              'generated_spike_train_random_glif_3_model_t_300s_rate_0_6.mat',
                              'generated_spike_train_glif_slower_rate_async_t_300s_rate_0_6.mat',
                              'generated_spike_train_random_glif_slower_more_synchronous_model_t_300s_rate_0_6.mat']
    output_fnames_rate_0_4 = []
    target_params_rate_0_6 = []
    target_params_rate_0_4 = []
    for fn in output_fnames_rate_0_6:
        output_fnames_rate_0_4.append(fn.replace('_6', '_4'))
        target_params_rate_0_6.append(fn.replace('.mat', '_params.pt'))
        target_params_rate_0_4.append(fn.replace('_6.mat', '_4_params.pt'))

    model_num = 0; rate_num = 0
    time_interval = 4000

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]
    for i, opt in enumerate(opts):
        if opt == '-h':
            print('run_single_objective_network_optim.py -mn <model-number> -rn <rate-number>')
            sys.exit()
        elif opt in ("-mn", "--model-number"):
            model_num = int(args[i])
        elif opt in ("-rn", "--rate-number"):
            rate_num = int(args[i])
    if rate_num == 0:
        output_fnames = output_fnames_rate_0_6
    elif rate_num == 1:
        output_fnames = output_fnames_rate_0_4
    else:
        raise NotImplementedError()

    for cur_output_fname in output_fnames:
        target_data_path = data_util.prefix + data_util.path
        output_data_path = target_data_path + cur_output_fname
        spike_node_indices, spike_times, spike_indices = data_util.load_sparse_data(output_data_path)
        # in_node_indices, input_times, input_indices = data_util.load_sparse_data(input_data_path)
        # _, sample_targets = data_util.get_spike_train_matrix(index_last_step=0, advance_by_t_steps=time_interval,
        #                                                      spike_times=spike_times, spike_indices=spike_indices,
        #                                                      node_numbers=spike_node_indices)
        target_params_dict = torch.load(target_data_path + target_params_rate_0_6[model_num])
        target_parameters = {}
        index_ctr = 0
        for param_i, key in enumerate(target_params_dict):
            if key not in ['loss_fn', 'rate', 'w']:
                target_parameters[index_ctr] = [target_params_dict[key].clone().detach().numpy()]
                index_ctr += 1
        # print('target_parameters:', target_parameters)

        # num_exps = 20; budget = 1000
        num_exps = 4; budget = 2
        params_by_optim = {}
        optim_names = ['PSO', 'NGO']; optim_ctr = 0
        for optim in [ng.optimizers.PSO, ng.optimizers.NGO]:
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


if __name__ == "__main__":
    main(sys.argv[1:])
