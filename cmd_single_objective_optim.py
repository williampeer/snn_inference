import nevergrad as ng

import IO
from Dev.brian2_custom_network_opt import *
from Log import Logger
from experiments import zip_dicts
from plot import plot_all_param_pairs_with_variance, plot_spiketrains_side_by_side

output_fnames_rate_0_6 = ['generated_spike_train_random_glif_1_model_t_300s_rate_0_6.mat',
                          'generated_spike_train_random_glif_2_model_t_300s_rate_0_6.mat',
                          'generated_spike_train_random_glif_3_model_t_300s_rate_0_6.mat',
                          'generated_spike_train_glif_slower_rate_async_t_300s_rate_0_6.mat',
                          'generated_spike_train_glif_slower_more_synchronous_model_t_300s_rate_0_6.mat']
output_fnames_rate_0_4 = []
target_params_rate_0_6 = []
target_params_rate_0_4 = []
for fn in output_fnames_rate_0_6:
    output_fnames_rate_0_4.append(fn.replace('_6', '_4'))
    target_params_rate_0_6.append(fn.replace('.mat', '_params.pt'))
    target_params_rate_0_4.append(fn.replace('_6.mat', '_4_params.pt'))

all_output_fnames = output_fnames_rate_0_6
for fn in target_params_rate_0_4:
    all_output_fnames.append(fn)

all_target_fnames = target_params_rate_0_6
for fn in target_params_rate_0_4:
    all_target_fnames.append(fn)


def main(argv):
    print('Argument List:', str(argv))

    num_exps = 20; budget = 10000
    # num_exps = 2; budget = 1
    optim_name = 'CMA'
    loss_fn = 'poisson_nll'
    time_interval = 4000

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]
    for i, opt in enumerate(opts):
        if opt == '-h':
            print('run_single_objective_network_optim.py -b <budget> -ne <num-experiments> -o <optim>')
            sys.exit()
        elif opt in ("-b", "--budget"):
            budget = int(args[i])
        elif opt in ("-ne", "--num-experiments"):
            num_exps = int(args[i])
        elif opt in ("-o", "--optim"):
            optim_name = args[i]
        elif opt in ("-lfn", "--loss-function"):
            loss_fn = args[i]

    if optim_name == 'DE':
        optim = ng.optimizers.DE
    elif optim_name == 'CMA':
        optim = ng.optimizers.CMA
    elif optim_name == 'PSO':
        optim = ng.optimizers.PSO
    elif optim_name == 'NGO':
        optim = ng.optimizers.NGO
    else:
        raise NotImplementedError()

    target_params_ctr = 0
    for model_filename in all_output_fnames:
        target_data_path = data_util.prefix + data_util.path
        output_data_path = target_data_path + model_filename

        spike_node_indices, spike_times, spike_indices = data_util.load_sparse_data(output_data_path)

        target_params_dict = torch.load(target_data_path + all_target_fnames[target_params_ctr])
        target_params_ctr += 1
        target_parameters = {}
        index_ctr = 0
        for param_i, key in enumerate(target_params_dict):
            if key not in ['loss_fn', 'rate', 'w']:
                target_parameters[index_ctr] = [target_params_dict[key].clone().detach().numpy()]
                index_ctr += 1

        # --------------------
        params_by_optim = {}
        UUID = IO.dt_descriptor()
        current_plottable_params_for_optim = {}
        other_params_for_optim = {}
        min_loss_per_exp = []
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

                                           loss_fn=loss_fn, spike_times=spike_times, spike_indices=spike_indices,
                                           spike_node_indices=spike_node_indices)

            optimizer = optim(parametrization=instrum, budget=budget)

            logger = Logger(log_fname='brian2_network_nevergrad_optimization_budget_{}'.format(budget))
            logger.log('setup experiment with the optimizer {}'.format(optimizer.__str__()))

            recommendation = optimizer.minimize(run_simulation_for)

            recommended_params = recommendation.value[1]
            del recommended_params['spike_times'], recommended_params['spike_indices'], recommended_params['spike_node_indices']

            logger.log('recommendation.value: {}'.format(recommended_params))

            cur_plot_params = {}
            min_loss_per_exp.append(recommendation.loss)
            index_ctr = 0
            for p_i, key in enumerate(recommended_params):
                if key in ['spike_times', 'spike_indices', 'spike_node_indices']:
                    pass
                elif key not in ['loss_fn', 'rate', 'w']:
                    if exp_i == 0:
                        current_plottable_params_for_optim[index_ctr] = [np.copy(recommended_params[key])]
                    else:
                        current_plottable_params_for_optim[index_ctr].append(np.copy(recommended_params[key]))
                    cur_plot_params[key] = np.copy(recommended_params[key])
                    index_ctr += 1
                else:
                    if exp_i == 0:
                        other_params_for_optim[key] = [np.copy(recommended_params[key])]
                    else:
                        other_params_for_optim[key].append(np.copy(recommended_params[key]))

            model_spike_train = get_spike_train_for(recommended_params['rate'], other_params_for_optim['w'][exp_i], cur_plot_params.copy())

            _, targets = data_util.get_spike_train_matrix(
                index_last_step=int(0.6 * np.random.rand() * spike_times.shape[0]),
                advance_by_t_steps=time_interval, spike_times=spike_times,
                spike_indices=spike_indices, node_numbers=spike_node_indices)
            plot_spiketrains_side_by_side(model_spike_train, targets, exp_type='single_objective_optim', uuid=UUID,
                                          title='Spike trains model and target ({}, loss: {})'.format(optim_name, recommendation.loss),
                                          fname='spike_trains_optim_{}_exp_num_{}'.format(optim_name, exp_i))

            torch.save(recommended_params.copy(),
                       './saved/single_objective_optim/fitted_params_optim_{}_loss_fn_{}_budget_{}_exp_{}.pt'.format(
                           optim_name, loss_fn, budget, exp_i))

        params_by_optim[optim_name] = zip_dicts(current_plottable_params_for_optim, other_params_for_optim)
        torch.save(params_by_optim, './saved/single_objective_optim/params_by_optim_{}_loss_fn_{}_budget_{}.pt'.format(optim_name, loss_fn, budget))
        torch.save(min_loss_per_exp, './saved/single_objective_optim/min_losses_optim_{}_loss_fn_{}_budget_{}.pt'.format(optim_name, loss_fn, budget))

        plot_all_param_pairs_with_variance(current_plottable_params_for_optim,
                                           exp_type='single_objective_optim',
                                           uuid=UUID,
                                           target_params=target_parameters,
                                           param_names=list(recommended_params.keys())[2:],
                                           custom_title="KDE projection of 2D model parameter".format(optim_name),
                                           logger=logger, fname='single_objective_KDE_optim_{}'.format(optim_name))


if __name__ == "__main__":
    main(sys.argv[1:])
