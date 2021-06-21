import sys

import torch
from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference.base import infer

import IO
from Models.no_grad.LIF_no_grad import LIF_no_grad
from TargetModels.TargetModels import lif_continuous_ensembles_model_dales_compliant
from experiments import poisson_input
from model_util import feed_inputs_sequentially_return_spike_train

torch.autograd.set_detect_anomaly(True)

# data_path = data_util.prefix + data_util.path + 'target_model_spikes_GLIF_seed_4_N_3_duration_300000.mat'
# node_indices, spike_times, spike_indices = data_util.load_sparse_data(full_path=data_path)
# next_step, targets = data_util.get_spike_train_matrix(index_last_step=0, advance_by_t_steps=t_interval,
#                                                       spike_times=spike_times, spike_indices=spike_indices, node_numbers=node_indices)
t_interval = 16000
N = 3


def main(argv):
    # methods = ['SNPE', 'SNLE', 'SNRE']
    # methods = ['SNPE']
    method = None

    print('Argument List:', str(argv))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]
    for i, opt in enumerate(opts):
        if opt == '-h':
            print('main.py -m <method>')
            sys.exit()
        elif opt in ("-m", "--method"):
            method = str(args[i])
        # elif opt in ("-N", "--num-neurons"):
        #     num_neurons = int(args[i])

    if method is not None:
        return sbi(method)


def sbi(method):
    num_dim = 1 + 3 * N + N ** 2

    tar_in_rate = 10.
    tar_model = lif_continuous_ensembles_model_dales_compliant(random_seed=0, N=N)
    inputs = poisson_input(rate=tar_in_rate, t=t_interval, N=N)
    parsed_weights = torch.zeros((N**2-N,))
    # for w_i in range(N**2-N):
    #     n_i = int((w_i) / N)
    #     n_j = w_i % N
    #     if n_j >= n_i:
    #         n_j = (n_j + 1) % N
    #         if n_j == 0:
    #             n_i = n_i+1
    #
    #     parsed_weights[w_i] = tar_model.w[n_i, n_j]
    ctr = 0
    for n_i in range(N):
        for n_j in range(N):
            if (n_i != n_j):
                parsed_weights[ctr] = tar_model.w[n_i, n_j]
                ctr += 1
    tar_parameters = torch.hstack([torch.tensor([tar_in_rate]), tar_model.E_L.data, tar_model.tau_m.data, tar_model.tau_s.data, parsed_weights])

    targets = feed_inputs_sequentially_return_spike_train(model=tar_model, inputs=inputs).clone().detach()

    # TODO: Programmatically create prior given model init params
    #   parameter_init_intervals = {'E_L': [-60., -60.], 'tau_m': [1.6, 1.6], 'tau_s': [2.5, 2.5]}
    # prior = utils.BoxUniform(low=-2*torch.ones(num_dim), high=2*torch.ones(num_dim))
    weights_low = torch.zeros((N**2-N,))
    weights_high = torch.ones((N**2-N,))
    # limits_low = torch.hstack((torch.tensor([2., -70., 1.6, 2.]), weights_low))
    # limits_high = torch.hstack((torch.tensor([20., -42., 3.0, 5.0]), weights_high))
    limits_low = torch.hstack((torch.tensor([4.]), torch.ones((N,))*-80., torch.ones((N,))*1.5, torch.ones((N,))*1.5, weights_low))
    limits_high = torch.hstack((torch.tensor([20.]), torch.ones((N,))*-35., torch.ones((N,))*8.0, torch.ones((N,))*10.0, weights_high))
    prior = utils.BoxUniform(low=limits_low, high=limits_high)

    res = {}
    # posterior_snpe = infer(simulator, prior, method='SNPE', num_simulations=5000)
    # posterior_snle = infer(simulator, prior, method='SNLE', num_simulations=5000)
    # posterior_snre = infer(simulator, prior, method='SNRE', num_simulations=5000)
    # posterior_stats(posterior_snpe, method='SNPE')
    # posterior_stats(posterior_snle, method='SNLE')
    # posterior_stats(posterior_snre, method='SNRE')

    # for m in methods:
    posterior = infer(LIF_simulator, prior, method=method, num_simulations=10000)
    # posterior = infer(LIF_simulator, prior, method=method, num_simulations=10)
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


def LIF_simulator(parameter_set):
    parsed_preset_weights = parameter_set[(1 + 3 * N):]
    assert len(parsed_preset_weights) == (N**2-N), "len(parsed_preset_weights): {}, should be N**2-N".format(len(parsed_preset_weights))
    preset_weights = torch.zeros((N, N))
    # for n_ctr in range(N**2-N):
    #     n_i = int(n_ctr/N)
    #     n_j = n_ctr % N
    #     if n_j >= n_i:
    #         n_j = n_j + 1
    #
    #     preset_weights[n_i, n_j] = parsed_preset_weights[n_ctr]
    ctr = 0
    for n_i in range(N):
        for n_j in range(N):
            if (n_i != n_j):
                preset_weights[n_i, n_j] = parsed_preset_weights[ctr]
                ctr += 1

    # print('preset_weights: {}'.format(preset_weights))
    params = {'E_L': parameter_set[1:(1+N)], 'tau_m': parameter_set[(1+N):(1+2*N)], 'tau_s': parameter_set[(1+2*N):(1+3*N)],
              'preset_weights': preset_weights}
    model = LIF_no_grad(parameters=params, N=N, neuron_types=[1, 1, -1])  # TODO: Auto-assign neuron-types for varying N != 12
    inputs = poisson_input(rate=parameter_set[0], t=t_interval, N=N)
    outputs = feed_inputs_sequentially_return_spike_train(model=model, inputs=inputs)
    model.reset()
    return outputs


def posterior_stats(posterior, method, observation, points, limits, figsize):
    print('====== def posterior_stats(posterior, method=None): =====')
    print(posterior)

    # observation = torch.reshape(targets, (1, -1))
    samples = posterior.sample((10000,), x=observation)
    # samples = posterior.sample((10,), x=observation)
    # log_probability = posterior.log_prob(samples, x=observation)
    try:
        fig, ax = analysis.pairplot(samples, points=points, limits=limits, figsize=figsize)
        if method is None:
            method = IO.dt_descriptor()
        fig.savefig('./figures/analysis_pairplot_{}.png'.format(method))
    except Exception as e:
        print("except: {}".format(e))


if __name__ == "__main__":
    res = main(sys.argv[1:])
