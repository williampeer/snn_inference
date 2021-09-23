import sys

import torch
from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference.base import infer

import IO
from Models.no_grad.GLIF_no_grad import GLIF_no_grad
from Models.no_grad.LIF_R_ASC_no_grad import LIF_R_ASC_no_grad
from Models.no_grad.LIF_R_no_grad import LIF_R_no_grad
from Models.no_grad.LIF_no_grad import LIF_no_grad
from TargetModels.TargetModels import lif_continuous_ensembles_model_dales_compliant, \
    lif_r_continuous_ensembles_model_dales_compliant, lif_r_asc_continuous_ensembles_model_dales_compliant, \
    glif_continuous_ensembles_model_dales_compliant
from experiments import sine_modulated_white_noise_input
from model_util import feed_inputs_sequentially_return_spike_train

torch.autograd.set_detect_anomaly(True)

# data_path = data_util.prefix + data_util.path + 'target_model_spikes_GLIF_seed_4_N_3_duration_300000.mat'
# node_indices, spike_times, spike_indices = data_util.load_sparse_data(full_path=data_path)
# next_step, targets = data_util.get_spike_train_matrix(index_last_step=0, advance_by_t_steps=t_interval,
#                                                       spike_times=spike_times, spike_indices=spike_indices, node_numbers=node_indices)


def main(argv):
    t_interval = 8000
    N = 3
    # methods = ['SNPE', 'SNLE', 'SNRE']
    # methods = ['SNPE']
    # method = None
    method = 'SNPE'
    model_type = 'LIF_R'

    print('Argument List:', str(argv))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]
    for i, opt in enumerate(opts):
        if opt == '-h':
            print('main.py -m <method> -mt <model-type> -N <num-neurons> -t <t-interval>')
            sys.exit()
        elif opt in ("-m", "--method"):
            method = str(args[i])
        elif opt in ("-mt", "--model-type"):
            model_type = str(args[i])
        elif opt in ("-N", "--num-neurons"):
            N = int(args[i])
        elif opt in ("-t", "--t-interval"):
            t_interval = int(args[i])

    if method is not None:
        return sbi(method, t_interval, N, model_type=model_type)


def sbi(method, t_interval, N, model_type = 'LIF'):
    lookup = {'LIF': LIF_no_grad, 'LIF_R': LIF_R_no_grad, 'LIF_R_ASC': LIF_R_ASC_no_grad, 'GLIF': GLIF_no_grad}
    m_class = lookup[model_type]

    num_params = len(m_class.parameter_names)

    def simulator(parameter_set):
        parsed_preset_weights = parameter_set[(1 + (num_params-1) * N):]
        assert len(parsed_preset_weights) == (N ** 2 - N), "len(parsed_preset_weights): {}, should be N**2-N".format(
            len(parsed_preset_weights))

        preset_weights = torch.zeros((N, N))
        ctr = 0
        for n_i in range(N):
            for n_j in range(N):
                if (n_i != n_j):
                    preset_weights[n_i, n_j] = parsed_preset_weights[ctr]
                    ctr += 1

        # print('preset_weights: {}'.format(preset_weights))
        params = {'E_L': parameter_set[1:(1 + N)], 'tau_m': parameter_set[(1 + N):(1 + 2 * N)],
                  'tau_s': parameter_set[(1 + 2 * N):(1 + 3 * N)],
                  'G': parameter_set[(1 + 3 * N):(1 + 4 * N)],
                  'f_v': parameter_set[(1 + 4 * N):(1 + 5 * N)],
                  'delta_theta_s': parameter_set[(1 + 5 * N):(1 + 6 * N)],
                  'b_s': parameter_set[(1 + 6 * N):(1 + 7 * N)],
                  'delta_V': parameter_set[(1 + 7 * N):(1 + 8 * N)],
                  'preset_weights': preset_weights}
        programmatic_neuron_types = torch.ones((N,))
        for n_i in range(int(2 * N / 3), N):
            programmatic_neuron_types[n_i] = -1

        model = m_class(parameters=params, N=N, neuron_types=programmatic_neuron_types)

        inputs = sine_modulated_white_noise_input(rate=parameter_set[0], t=t_interval, N=N)
        outputs = feed_inputs_sequentially_return_spike_train(model=model, inputs=inputs)
        model.reset()
        return outputs

    # num_dim = 1 + 3 * N + N ** 2
    num_dim = 1 + (num_params-1) * N + (N ** 2 - N)

    tar_in_rate = 10.
    lookup_tar_m_fn = { 'LIF': lif_continuous_ensembles_model_dales_compliant,
                      'LIF_R': lif_r_continuous_ensembles_model_dales_compliant,
                      'LIF_R_ASC': lif_r_asc_continuous_ensembles_model_dales_compliant,
                      'GLIF': glif_continuous_ensembles_model_dales_compliant }
    tar_m_fn = lookup_tar_m_fn[model_type]
    tar_model = tar_m_fn(random_seed=42, N=N)
    inputs = sine_modulated_white_noise_input(rate=tar_in_rate, t=t_interval, N=N)
    parsed_weights = torch.zeros((N**2-N,))
    ctr = 0
    for n_i in range(N):
        for n_j in range(N):
            if (n_i != n_j):
                parsed_weights[ctr] = abs(tar_model.w[n_i, n_j])
                ctr += 1

    # parameter_names = ['w', 'E_L', 'tau_m', 'G', 'f_v', 'f_I', 'delta_theta_s', 'b_s', 'a_v', 'b_v', 'theta_inf', 'delta_V', 'tau_s']
    tar_params_iter = torch.hstack([tar_model.E_L, tar_model.tau_m, tar_model.tau_s, tar_model.G, tar_model.f_v, tar_model.delta_theta_s, tar_model.b_s, tar_model.delta_V])
    tar_parameters = torch.hstack([torch.tensor([tar_in_rate]), tar_params_iter, parsed_weights])

    targets = feed_inputs_sequentially_return_spike_train(model=tar_model, inputs=inputs).clone().detach()

    # TODO: Programmatically create prior given model init params
    #   parameter_init_intervals = {'E_L': [-60., -60.], 'tau_m': [1.6, 1.6], 'tau_s': [2.5, 2.5]}
    # prior = utils.BoxUniform(low=-2*torch.ones(num_dim), high=2*torch.ones(num_dim))
    weights_low = torch.zeros((N**2-N,))
    weights_high = torch.ones((N**2-N,))
    # limits_low = torch.hstack((torch.tensor([2., -70., 1.6, 2.]), weights_low))
    # limits_high = torch.hstack((torch.tensor([20., -42., 3.0, 5.0]), weights_high))
    limits_low = torch.hstack((torch.tensor([4.]), torch.ones((N,))*-80., torch.ones((N,))*1.5, torch.ones((N,))*1.5,
                               0.01*torch.ones((N,)), 0.01*torch.ones((N,)), 6.*torch.ones((N,)), torch.ones((N,)), 0.01*torch.ones((N,)),
                               weights_low))
    limits_high = torch.hstack((torch.tensor([20.]), torch.ones((N,))*-35., torch.ones((N,))*8.0, torch.ones((N,))*10.0,
                                0.99*torch.ones((N,)), 0.99*torch.ones((N,)), 30.*torch.ones((N,)), 35.*torch.ones((N,)), 0.99*torch.ones((N,)),
                                weights_high))
    prior = utils.BoxUniform(low=limits_low, high=limits_high)

    res = {}
    # posterior_snpe = infer(simulator, prior, method='SNPE', num_simulations=5000)
    # posterior_snle = infer(simulator, prior, method='SNLE', num_simulations=5000)
    # posterior_snre = infer(simulator, prior, method='SNRE', num_simulations=5000)
    # posterior_stats(posterior_snpe, method='SNPE')
    # posterior_stats(posterior_snle, method='SNLE')
    # posterior_stats(posterior_snre, method='SNRE')

    # for m in methods:
    posterior = infer(simulator, prior, method=method, num_simulations=8000)
    # posterior = infer(simulator, prior, method=method, num_simulations=10)
    res[method] = posterior
    posterior_stats(posterior, method=method, observation=torch.reshape(targets, (1, -1)), points=tar_parameters,
                    limits=torch.stack((limits_low, limits_high), dim=1), figsize=(num_dim, num_dim))

    try:
        dt_descr = IO.dt_descriptor()
        IO.save_data(res, 'sbi_res', description='Res from SBI using {}, dt descr: {}'.format(method, dt_descr),
                     fname='res_{}_dt_{}'.format(method, dt_descr))
    except Exception as e:
        print("except: {}".format(e))

    return res


def posterior_stats(posterior, method, observation, points, limits, figsize):
    print('====== def posterior_stats(posterior, method=None): =====')
    print(posterior)

    # observation = torch.reshape(targets, (1, -1))
    samples = posterior.sample((6000,), x=observation)
    # samples = posterior.sample((10,), x=observation)
    # log_probability = posterior.log_prob(samples, x=observation)
    try:
        fig, ax = analysis.pairplot(samples, points=points, limits=limits, figsize=figsize)
        if method is None:
            method = IO.dt_descriptor()
        fig.savefig('./figures/analysis_pairplot_{}_{}.png'.format(method, IO.dt_descriptor()))
    except Exception as e:
        print("except: {}".format(e))


if __name__ == "__main__":
    res = main(sys.argv[1:])
    sys.exit(0)
