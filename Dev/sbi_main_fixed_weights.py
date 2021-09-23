import sys

import torch
from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference.base import infer

import IO
from Models.no_grad.LIF_no_grad import LIF_no_grad
from TargetModels.TargetModels import lif_continuous_ensembles_model_dales_compliant
from experiments import sine_modulated_white_noise_input
from model_util import feed_inputs_sequentially_return_spike_train

torch.autograd.set_detect_anomaly(True)

# data_path = data_util.prefix + data_util.path + 'target_model_spikes_GLIF_seed_4_N_3_duration_300000.mat'
# node_indices, spike_times, spike_indices = data_util.load_sparse_data(full_path=data_path)
# next_step, targets = data_util.get_spike_train_matrix(index_last_step=0, advance_by_t_steps=t_interval,
#                                                       spike_times=spike_times, spike_indices=spike_indices, node_numbers=node_indices)


def main(argv):
    t_interval = 16000
    N = 3
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
        elif opt in ("-N", "--num-neurons"):
            N = int(args[i])
        elif opt in ("-t", "--t-interval"):
            t_interval = int(args[i])

    if method is not None:
        return sbi(method, t_interval, N)


def sbi(method, t_interval, N):
    def LIF_simulator(parameter_set):
        tar_model = lif_continuous_ensembles_model_dales_compliant(random_seed=42, N=N)
        # print('preset_weights: {}'.format(preset_weights))
        params = {'E_L': parameter_set[1:(1 + N)], 'tau_m': parameter_set[(1 + N):(1 + 2 * N)],
                  'tau_s': parameter_set[(1 + 2 * N):(1 + 3 * N)],
                  'preset_weights': tar_model.w.clone().detach()}
        programmatic_neuron_types = torch.ones((N,))
        for n_i in range(int(2 * N / 3), N):
            programmatic_neuron_types[n_i] = -1
        model = LIF_no_grad(parameters=params, N=N, neuron_types=programmatic_neuron_types)
        inputs = sine_modulated_white_noise_input(rate=parameter_set[0], t=t_interval, N=N)
        outputs = feed_inputs_sequentially_return_spike_train(model=model, inputs=inputs)
        model.reset()
        return outputs

    num_dim = 1 + 3 * N + N ** 2

    tar_in_rate = 10.
    tar_model = lif_continuous_ensembles_model_dales_compliant(random_seed=42, N=N)
    inputs = sine_modulated_white_noise_input(rate=tar_in_rate, t=t_interval, N=N)

    tar_parameters = torch.hstack([torch.tensor([tar_in_rate]), tar_model.E_L.data, tar_model.tau_m.data, tar_model.tau_s.data])

    targets = feed_inputs_sequentially_return_spike_train(model=tar_model, inputs=inputs).clone().detach()

    limits_low = torch.hstack((torch.tensor([4.]), torch.ones((N,))*-80., torch.ones((N,))*1.5, torch.ones((N,))*1.5))
    limits_high = torch.hstack((torch.tensor([20.]), torch.ones((N,))*-35., torch.ones((N,))*8.0, torch.ones((N,))*10.0))
    prior = utils.BoxUniform(low=limits_low, high=limits_high)

    res = {}

    # for m in methods:
    posterior = infer(LIF_simulator, prior, method=method, num_simulations=10000)
    # posterior = infer(LIF_simulator, prior, method=method, num_simulations=20)
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
    samples = posterior.sample((10000,), x=observation)
    # samples = posterior.sample((20,), x=observation)
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
