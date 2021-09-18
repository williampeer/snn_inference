import sys

import torch
from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference.base import infer

import IO
from Models.no_grad.GLIF_soft_no_grad import GLIF_soft_no_grad
from Models.no_grad.LIF_R_ASC_no_grad import LIF_R_ASC_no_grad
from Models.no_grad.LIF_R_soft_no_grad import LIF_R_soft_no_grad
from TargetModels.TargetModels import lif_r_asc_continuous_ensembles_model_dales_compliant
from TargetModels.TargetModelsSoft import lif_r_soft_continuous_ensembles_model_dales_compliant, \
    glif_soft_continuous_ensembles_model_dales_compliant
from analysis.sbi_export_plots import export_plots
from experiments import poisson_input
from model_util import feed_inputs_sequentially_return_spike_train

torch.autograd.set_detect_anomaly(True)

# data_path = data_util.prefix + data_util.path + 'target_model_spikes_GLIF_seed_4_N_3_duration_300000.mat'
# node_indices, spike_times, spike_indices = data_util.load_sparse_data(full_path=data_path)
# next_step, targets = data_util.get_spike_train_matrix(index_last_step=0, advance_by_t_steps=t_interval,
#                                                       spike_times=spike_times, spike_indices=spike_indices, node_numbers=node_indices)


def transform_model_to_sbi_params(model):
    m_params = torch.zeros((model.N**2-model.N,))
    ctr = 0
    for i in range(model.w.shape[0]):
        for j in range(model.w.shape[1]):
            if i!=j:
                m_params[ctr] = model.w[i,j].clone().detach()
                ctr += 1

    model_params_list = model.get_parameters()
    for p_i in range(1, len(model.__class__.parameter_names)):
        m_params = torch.hstack((m_params, model_params_list[p_i]))
        # model_params_list[(N ** 2 - N) + N * (i - 1):(N ** 2 - N) + N * i] = [model_class.parameter_names[i]]

    return m_params


def main(argv):
    NUM_WORKERS = 16

    t_interval = 16000
    N = 3
    # methods = ['SNPE', 'SNLE', 'SNRE']
    # methods = ['SNPE']
    # method = None
    method = 'SNRE'
    # model_type = None
    model_type = 'LIF_R_soft'
    budget = 10000
    # budget = 40
    tar_seed = 42

    # class_lookup = { 'LIF': LIF_no_grad, 'LIF_R': LIF_R_no_grad, 'LIF_R_ASC': LIF_R_ASC_no_grad, 'GLIF': GLIF_no_grad }
    class_lookup = { 'LIF_R_soft': LIF_R_soft_no_grad, 'LIF_R_ASC': LIF_R_ASC_no_grad, 'GLIF_soft': GLIF_soft_no_grad }

    print('Argument List:', str(argv))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]
    for i, opt in enumerate(opts):
        if opt == '-h':
            print('main.py -m <method> -N <num-neurons> -t <t-interval> -pn <param-number> -b <budget> -nw <num-workers>')
            sys.exit()
        elif opt in ("-m", "--method"):
            method = str(args[i])
        elif opt in ("-mt", "--model-type"):
            model_type = str(args[i])
        elif opt in ("-N", "--num-neurons"):
            N = int(args[i])
        elif opt in ("-t", "--t-interval"):
            t_interval = int(args[i])
        elif opt in ("-pn", "--param-number"):
            param_number = int(args[i])
        elif opt in ("-b", "--budget"):
            budget = int(args[i])
        elif opt in ("-nw", "--num-workers"):
            NUM_WORKERS = int(args[i])
        elif opt in ("-ts", "--tar-seed"):
            tar_seed = int(args[i])

    # assert param_number >= 0, "please specify a parameter to fit. (-pn || --param-number)"
    assert model_type is not None, "please specify a model type (-mt || --model-type)"
    model_class = class_lookup[model_type]
    # assert param_number < len(model_class.parameter_names), \
    #     "param_number: {} cannot be greater than number of parameters: {} in model_class: {}" \
    #         .format(param_number, len(model_class.parameter_names), model_class)

    if method is not None:
        sbi(method, t_interval, N, model_class, budget, tar_seed, NUM_WORKERS)


# def get_spike_rates(out, bins=10):
#     bin_len = int(out.shape[0] / bins)
#     out_counts = torch.zeros((bins, out.shape[1]))
#     for b_i in range(bins):
#         out_counts[b_i] = (out[b_i * bin_len:(b_i + 1) * bin_len].sum(dim=0))
#     return out_counts


def sbi(method, t_interval, N, model_class, budget, tar_seed, NUM_WORKERS=6):
    tar_model_fn_lookup = { 'LIF_R_soft_no_grad': lif_r_soft_continuous_ensembles_model_dales_compliant,
                            'LIF_R_ASC_no_grad': lif_r_asc_continuous_ensembles_model_dales_compliant,
                            'GLIF_soft_no_grad': glif_soft_continuous_ensembles_model_dales_compliant }
    tar_in_rate = 10.
    tar_model_fn = tar_model_fn_lookup[model_class.__name__]
    tar_model = tar_model_fn(random_seed=tar_seed, N=N)

    def simulator(parameter_set):
        programmatic_params_dict = {}
        parsed_preset_weights = parameter_set[:(N**2-N)]
        assert len(parsed_preset_weights) == (N ** 2 - N), "len(parsed_preset_weights): {}, should be N**2-N".format(
            len(parsed_preset_weights))
        preset_weights = torch.zeros((N, N))
        ctr = 0
        for n_i in range(N):
            for n_j in range(N):
                if (n_i != n_j):
                    preset_weights[n_i, n_j] = parsed_preset_weights[ctr]
                    ctr += 1
        programmatic_params_dict[model_class.parameter_names[0]] = preset_weights

        for i in range(1, len(model_class.parameter_names)):
            programmatic_params_dict[model_class.parameter_names[i]] = parameter_set[(N**2-N)+N*(i-1):(N**2-N)+N*i]  # assuming only N-dimensional params otherwise

        programmatic_neuron_types = torch.ones((N,))
        for n_i in range(int(2 * N / 3), N):
            programmatic_neuron_types[n_i] = -1

        model = model_class(parameters=programmatic_params_dict, N=N, neuron_types=programmatic_neuron_types)
        inputs = poisson_input(rate=tar_in_rate, t=t_interval, N=N)
        outputs = feed_inputs_sequentially_return_spike_train(model=model, inputs=inputs)

        model.reset()
        mean_output_rates = outputs.sum(dim=0) * 1000. / outputs.shape[0]  # Hz
        return mean_output_rates

        # return torch.reshape(get_binned_spike_counts(outputs.clone().detach()), (-1,))

        # return outputs.clone().detach()

    # inputs = poisson_input(rate=tar_in_rate, t=t_interval, N=N)

    limits_low = torch.zeros((N**2-N,))
    limits_high = torch.ones((N**2-N,))

    for i in range(1, len(model_class.parameter_names)):
        limits_low = torch.hstack((limits_low, torch.ones((N,)) * model_class.param_lin_constraints[i][0]))
        limits_high = torch.hstack((limits_high, torch.ones((N,)) * model_class.param_lin_constraints[i][1]))

    prior = utils.BoxUniform(low=limits_low, high=limits_high)

    tar_sbi_params = transform_model_to_sbi_params(tar_model)
    targets_per_sample = None
    n_samples = 8
    for i in range(n_samples):
        cur_targets = simulator(tar_sbi_params)
        if targets_per_sample is None:
            targets_per_sample = cur_targets
        else:
            # spike_counts_per_sample = torch.vstack((spike_counts_per_sample, cur_cur_spike_count))
            targets_per_sample = targets_per_sample + cur_targets
    avg_tar_model_simulations = targets_per_sample / n_samples

    posterior = infer(simulator, prior, method=method, num_simulations=budget, num_workers=NUM_WORKERS)
    # posterior = infer(LIF_simulator, prior, method=method, num_simulations=10)
    dt_descriptor = IO.dt_descriptor()
    res = {}
    res[method] = posterior
    res['model_class'] = model_class
    res['N'] = N
    res['dt_descriptor'] = dt_descriptor
    res['tar_seed'] = tar_seed
    # num_dim = N**2-N+N*(len(model_class.parameter_names)-1)
    num_dim = limits_high.shape[0]

    try:
        IO.save_data(res, 'sbi_res', description='Res from SBI using {}, dt descr: {}'.format(method, dt_descriptor),
                     fname='res_{}_dt_{}_tar_seed_{}'.format(method, dt_descriptor, tar_seed))

        posterior_stats(posterior, method=method,
                        # observation=torch.reshape(avg_tar_model_simulations, (-1, 1)), points=tar_sbi_params,
                        observation=avg_tar_model_simulations, points=tar_sbi_params,
                        limits=torch.stack((limits_low, limits_high), dim=1), figsize=(num_dim, num_dim), budget=budget,
                        m_name=tar_model.name(), dt_descriptor=dt_descriptor, tar_seed=tar_seed)
    except Exception as e:
        print("except: {}".format(e))

    return res


def posterior_stats(posterior, method, observation, points, limits, figsize, budget, m_name, dt_descriptor, tar_seed):
    print('====== def posterior_stats(posterior, method=None): =====')
    print(posterior)

    # observation = torch.reshape(targets, (1, -1))
    data_arr = {}
    samples = posterior.sample((budget,), x=observation)
    data_arr['samples'] = samples
    data_arr['observation'] = observation
    data_arr['tar_parameters'] = points
    data_arr['m_name'] = m_name

    # samples = posterior.sample((10,), x=observation)
    # log_probability = posterior.log_prob(samples, x=observation)
    # print('log_probability: {}'.format(log_probability))
    IO.save_data(data_arr, 'sbi_samples', description='Res from SBI using {}, dt descr: {}'.format(method, dt_descriptor),
                 fname='samples_method_{}_m_name_{}_dt_{}_tar_seed_{}'.format(method, m_name, dt_descriptor, tar_seed))

    # checking docs for convergence criterion
    # plot 100d
    try:
        # def export_plots(samples, points, lim_low, lim_high, N, method, m_name, description, model_class):
        export_plots(samples, points, limits[0], limits[1], len(points[1]), 'SNRE', m_name, 'sbi_export_{}'.format(dt_descriptor), m_name)
    except Exception as e:
        print('exception in new plot code: {}'.format(e))

    try:
        if samples[0].shape[0] <= 10:
            fig, ax = analysis.pairplot(samples, points=points, limits=limits, figsize=figsize)
            if method is None:
                method = dt_descriptor
            fig.savefig('./figures/analysis_pairplot_{}_one_param_{}_{}.png'.format(method, m_name, dt_descriptor))
    except Exception as e:
        print("except: {}".format(e))


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)