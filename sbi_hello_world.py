import sys

import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer

import IO
import data_util
from Models.no_grad.LIF_no_grad import LIF_no_grad
from experiments import poisson_input
from model_util import feed_inputs_sequentially_return_spike_train

torch.autograd.set_detect_anomaly(True)

t_interval = 10000
N = 3
num_dim = N**2 + 3*N

data_path = data_util.prefix + data_util.path + 'target_model_spikes_GLIF_seed_4_N_3_duration_300000.mat'
node_indices, spike_times, spike_indices = data_util.load_sparse_data(full_path=data_path)
next_step, targets = data_util.get_spike_train_matrix(index_last_step=0, advance_by_t_steps=t_interval,
                                                      spike_times=spike_times, spike_indices=spike_indices, node_numbers=node_indices)

# TODO: Programmatically create prior given model init params
#   parameter_init_intervals = {'E_L': [-60., -60.], 'tau_m': [1.6, 1.6], 'tau_s': [2.5, 2.5]}
# prior = utils.BoxUniform(low=-2*torch.ones(num_dim), high=2*torch.ones(num_dim))
weights_low = torch.flatten(torch.zeros((N, N)))
weights_high = torch.flatten(torch.ones((N, N)))
limits_low = torch.hstack((torch.tensor([2., -70., 1.6, 2.]), weights_low))
limits_high = torch.hstack((torch.tensor([20., -42., 3.0, 5.0]), weights_high))
prior = utils.BoxUniform(low=limits_low, high=limits_high)


def simulator(parameter_set):
    params = {'E_L': parameter_set[1], 'tau_m': parameter_set[2], 'tau_s': parameter_set[3],
              'preset_weights': torch.reshape(parameter_set[4:], (N, N))}
    model = LIF_no_grad(parameters=params, N=N, neuron_types=[1, 1, -1])  # TODO: Auto-assign neuron-types for varying N != 12
    inputs = poisson_input(rate=parameter_set[0], t=t_interval, N=N)
    outputs = feed_inputs_sequentially_return_spike_train(model=model, inputs=inputs)
    model.reset()
    return outputs


def posterior_stats(posterior, method=None):
    observation = torch.reshape(targets, (1, -1))
    samples = posterior.sample((10000,), x=observation)
    log_probability = posterior.log_prob(samples, x=observation)
    fig, ax = analysis.pairplot(samples, limits=torch.stack((limits_low, limits_high), dim=1), figsize=(num_dim, num_dim))
    if method is None:
        method = IO.dt_descriptor()
    fig.savefig('./figures/analysis_pairplot_{}.png'.format(method))

methods = ['SNPE', 'SNLE', 'SNRE']
res = {}
# posterior_snpe = infer(simulator, prior, method='SNPE', num_simulations=5000)
# posterior_snle = infer(simulator, prior, method='SNLE', num_simulations=5000)
# posterior_snre = infer(simulator, prior, method='SNRE', num_simulations=5000)
# posterior_stats(posterior_snpe, method='SNPE')
# posterior_stats(posterior_snle, method='SNLE')
# posterior_stats(posterior_snre, method='SNRE')
for m in methods:
    posterior = infer(simulator, prior, method=m, num_simulations=5000)
    res[m] = posterior
    posterior_stats(posterior, method=m)
