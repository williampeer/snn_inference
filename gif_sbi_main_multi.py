import sys

import torch
import torch.tensor as T
from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference.base import infer

import IO
from Models.LowerDim.GLIF_soft_lower_dim import GLIF_soft_lower_dim
from Models.LowerDim.LIF_R_soft_lower_dim import LIF_R_soft_lower_dim
from Models.microGIF import microGIF
from Models.no_grad.GLIF_no_grad import GLIF_no_grad
from Models.no_grad.LIF_R_ASC_no_grad import LIF_R_ASC_no_grad
from Models.no_grad.LIF_R_no_grad import LIF_R_no_grad
from TargetModels.TargetModelMicroGIF import micro_gif_populations_model
from TargetModels.TargetModelsSoft import glif_soft_continuous_ensembles_model_dales_compliant
from experiments import sine_modulated_white_noise
from model_util import feed_inputs_sequentially_return_tuple

torch.autograd.set_detect_anomaly(True)

# data_path = data_util.prefix + data_util.path + 'target_model_spikes_GLIF_seed_4_N_3_duration_300000.mat'
# node_indices, spike_times, spike_indices = data_util.load_sparse_data(full_path=data_path)
# next_step, targets = data_util.get_spike_train_matrix(index_last_step=0, advance_by_t_steps=t_interval,
#                                                       spike_times=spike_times, spike_indices=spike_indices, node_numbers=node_indices)


def transform_model_to_sbi_params(model, model_class):
    m_params = torch.zeros((model.N**2-model.N,))
    ctr = 0
    for i in range(model.w.shape[0]):
        for j in range(model.w.shape[1]):
            if i!=j:
                m_params[ctr] = model.w[i, j].clone().detach()
                ctr += 1

    model_params = model.get_parameters()
    for p_i, p_k in enumerate(model_params):
        if p_k is not 'w' and p_k in model_class.free_parameters:
            m_params = torch.hstack((m_params, model_params[p_k]))
        # model_params_list[(N ** 2 - N) + N * (i - 1):(N ** 2 - N) + N * i] = [model_class.free_parameters[i]]

    return m_params


def main(argv):
    NUM_WORKERS = 4
    # NUM_WORKERS = 1

    # t_interval = 10000
    t_interval = 1600
    N = 8
    # methods = ['SNPE', 'SNLE', 'SNRE']
    # methods = ['SNPE']
    # method = None
    method = 'SNPE'
    # model_type = None
    model_type = 'microGIF'
    # budget = 10000
    budget = 100
    tar_seed = 42

    class_lookup = { 'LIF_R': LIF_R_no_grad, 'LIF_R_ASC': LIF_R_ASC_no_grad, 'GLIF': GLIF_no_grad,
                     'GLIF_soft_lower_dim' : GLIF_soft_lower_dim, 'LIF_R_soft_lower_dim': LIF_R_soft_lower_dim,
                     'microGIF': microGIF }

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
        elif opt in ("-b", "--budget"):
            budget = int(args[i])
        elif opt in ("-nw", "--num-workers"):
            NUM_WORKERS = int(args[i])
        elif opt in ("-ts", "--tar-seed"):
            tar_seed = int(args[i])

    # assert param_number >= 0, "please specify a parameter to fit. (-pn || --param-number)"
    assert model_type is not None, "please specify a model type (-mt || --model-type)"
    model_class = class_lookup[model_type]

    if method is not None:
        sbi(method, t_interval, N, model_class, budget, tar_seed, NUM_WORKERS)


def get_binned_spike_counts(out, bin_size=200):
    # bin_len = int(out.shape[0] / bins)
    n_bins = int(out.shape[0] / bin_size)
    out_counts = torch.zeros((n_bins, out.shape[1]))
    for b_i in range(n_bins):
        out_counts[b_i] = (out[b_i * bin_size:(b_i + 1) * bin_size].sum(dim=0))
    return out_counts


def sbi(method, t_interval, N, model_class, budget, tar_seed, NUM_WORKERS=5):
    tar_model_fn_lookup = { 'GLIF_soft_lower_dim': glif_soft_continuous_ensembles_model_dales_compliant,
                            'microGIF': micro_gif_populations_model }
    # tar_in_rate = 10.
    tar_model_fn = tar_model_fn_lookup[model_class.__name__]
    if N == 4:
        N_pops = 2
        pop_size = 2
    elif N == 16:
        N_pops = 4
        pop_size = 4
    elif N == 8:
        N_pops = 4
        pop_size = 2
    elif N == 2:
        N_pops = 2
        pop_size = 1
    else:
        raise NotImplementedError('N has to be in [2, 4, 8, 16]')

    tar_model = tar_model_fn(random_seed=tar_seed, pop_size=pop_size, N_pops=N_pops)

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
        programmatic_params_dict[model_class.free_parameters[0]] = preset_weights

        tar_params = tar_model.get_parameters()
        for p_i, p_k in enumerate(tar_model.get_parameters()):
            if not model_class.free_parameters.__contains__(p_k):
                programmatic_params_dict[p_k] = tar_params[p_k].clone().detach()

        for i in range(1, len(model_class.free_parameters)):
            programmatic_params_dict[model_class.free_parameters[i]] = parameter_set[(N**2-N)+N*(i-1):(N**2-N)+N*i]  # assuming only N-dimensional params otherwise

        programmatic_neuron_types = torch.ones((N,))
        for n_i in range(int(N / 2), N):
            programmatic_neuron_types[n_i] = -1

        model = model_class(parameters=programmatic_params_dict, N=N, neuron_types=programmatic_neuron_types)
        inputs = sine_modulated_white_noise(t=t_interval, N=N, neurons_coeff = torch.cat([T(int(N / 2) * [0.]), T(int(N/4) * [0.25]), T(int(N/4) * [0.1])]))
        spike_probas, outputs = feed_inputs_sequentially_return_tuple(model=model, inputs=inputs)

        return torch.reshape(get_binned_spike_counts(outputs.clone().detach()), (-1,))

    limits_low = torch.zeros((N**2-N,))
    limits_high = torch.ones((N**2-N,))

    for i in range(1, len(model_class.free_parameters)):
        limits_low = torch.hstack((limits_low, torch.ones((N,)) * model_class.param_lin_constraints[i][0]))
        limits_high = torch.hstack((limits_high, torch.ones((N,)) * model_class.param_lin_constraints[i][1]))

    prior = utils.BoxUniform(low=limits_low, high=limits_high)

    tar_sbi_params = transform_model_to_sbi_params(tar_model, model_class)

    posterior = infer(simulator, prior, method=method, num_simulations=budget, num_workers=NUM_WORKERS)
    # posterior = infer(LIF_simulator, prior, method=method, num_simulations=10)
    dt_descriptor = IO.dt_descriptor()
    res = {}
    res[method] = posterior
    res['model_class'] = model_class
    res['N'] = N
    res['dt_descriptor'] = dt_descriptor
    res['tar_seed'] = tar_seed
    # num_dim = N**2-N+N*(len(model_class.free_parameters)-1)
    num_dim = limits_high.shape[0]

    # try:
    IO.save_data(res, 'sbi_res', description='Res from SBI using {}, dt descr: {}'.format(method, dt_descriptor),
                 fname='res_{}_dt_{}_tar_seed_{}'.format(method, dt_descriptor, tar_seed))

    targets = simulator(tar_sbi_params)
    posterior_stats(posterior, method=method,
                    # observation=torch.reshape(avg_tar_model_simulations, (-1, 1)), points=tar_sbi_params,
                    observation=targets, points=tar_sbi_params, model_dim=N, plot_dim=num_dim,
                    limits=torch.stack((limits_low, limits_high), dim=1), figsize=(num_dim, num_dim), budget=budget,
                    m_name=tar_model.name(), dt_descriptor=dt_descriptor, tar_seed=tar_seed, model_class=model_class)
    # except Exception as e:
    #     print("except: {}".format(e))

    return res


def posterior_stats(posterior, method, observation, points, model_dim, plot_dim, limits, figsize, budget,
                    m_name, dt_descriptor, tar_seed, model_class):
    print('====== def posterior_stats(posterior, method=None): =====')
    print(posterior)

    # observation = torch.reshape(targets, (1, -1))
    data_arr = {}
    samples = posterior.sample((budget,), x=observation, sample_with_mcmc=True)
    data_arr['samples'] = samples
    data_arr['observation'] = observation
    data_arr['tar_parameters'] = points
    data_arr['m_name'] = m_name

    # samples = posterior.sample((10,), x=observation)
    # log_probability = posterior.log_prob(samples, x=observation)
    # print('log_probability: {}'.format(log_probability))
    IO.save_data(data_arr, 'sbi_samples', description='Res from SBI using {}, dt descr: {}'.format(method, dt_descriptor),
                 fname='samples_method_{}_m_name_{}_dt_{}_tar_seed_{}'.format(method, m_name, dt_descriptor, tar_seed))

    plot_dim = len(points)
    export_plots(samples, points, limits, model_dim, plot_dim, method, m_name, 'sbi_export_{}'.format(dt_descriptor), model_class)
    # except Exception as e:
    #     print('exception in new plot code: {}'.format(e))
    #     print('samples: {}\npoints: {}\nlimits[0]: {}\nlimits[1]: {}\nmodel_dim: {}'.format(samples, points, limits[0], limits[1], model_dim))
    sys.exit(0)


def export_plots(samples, points, limits, model_dim, plot_dim, method, m_name, description, model_class):
    N = model_dim
    assert limits.shape[1] == 2, "limits.shape[0] should be 2. limits.shape: {}".format(limits.shape)
    lim_low = limits[:,0]
    lim_high = limits[:,1]
    # if plot_dim < 12:  # full marginal plot
    #     plt.figure()
    #     fig, ax = analysis.pairplot(samples, points=points, limits=limits, figsize=(plot_dim, plot_dim))
    #     fig.savefig('./figures/export_analysis_pairplot_{}_one_param_{}_{}.png'.format(method, m_name, description))
    #     plt.close()
    # else:
    # plt.figure()
    weights_offset = N ** 2 - N

    # WEIGHTS
    # plt.figure()
    # cur_limits = torch.stack((lim_low[:weights_offset], lim_high[:weights_offset]))
    cur_mean_limits = torch.stack((torch.zeros((N,)), torch.ones((N,))))
    cur_pt = points[:weights_offset]
    cur_samples = samples[:, :weights_offset]

    weights_mean = torch.tensor([])
    tar_ws_mean = torch.tensor([])
    # cur_limits_low_mean = torch.tensor([]); cur_limits_high_mean = torch.tensor([])
    for n_i in range(N):
        # for w_i in range(N-1):
        weights_mean = torch.hstack([weights_mean, torch.reshape(torch.mean(cur_samples[:, n_i*(N-1):(n_i+1)*(N-1)], axis=-1), (-1, 1))])
        tar_ws_mean = torch.hstack([tar_ws_mean, torch.reshape(torch.mean(cur_pt[n_i*(N-1):(n_i+1)*(N-1)], axis=-1), (-1, 1))])
        # cur_limits_mean = torch.cat([cur_limits_mean, torch.mean(cur_limits[:, n_i*N:n_i*N+(N-1)], axis=0)])

    fig_subset_mean, ax_mean = analysis.pairplot(weights_mean, points=tar_ws_mean, limits=cur_mean_limits.T, figsize=(N, N))
    path = './figures/sbi/{}/{}/'.format(m_name, description)
    IO.makedir_if_not_exists(path)
    fname = 'export_sut_subset_analysis_pairplot_{}_{}_weights_{}.png'.format(method, m_name, description)
    fig_subset_mean.savefig(path + fname)
    # plt.close()

    # Marginals only for p_i, p_i
    for p_i in range(1, len(model_class.free_parameters)):
        # plt.figure()
        cur_mean_limits = torch.stack((lim_low[weights_offset+(p_i-1)*N:weights_offset+p_i*N], lim_high[weights_offset+(p_i-1)*N:weights_offset+p_i*N]))
        cur_pt = points[weights_offset+(p_i-1)*N:weights_offset+p_i*N]
        cur_samples = samples[:, weights_offset+(p_i-1)*N:weights_offset+p_i*N]
        fig_subset_mean, ax_mean = analysis.pairplot(cur_samples, points=cur_pt, limits=cur_mean_limits.T, figsize=(N, N))
        path = './figures/sbi/{}/{}/'.format(m_name, description)
        IO.makedir_if_not_exists(path)
        fname = 'export_sut_subset_analysis_pairplot_{}_{}_one_param_{}_{}.png'.format(method, m_name, p_i, description)
        fig_subset_mean.savefig(path + fname)
        # plt.close()


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
