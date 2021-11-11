import sys

import numpy as np
import torch

import experiments
import model_util
import spike_metrics
from Models.microGIF import microGIF
from Models.microGIF_exact import microGIF_exact
from TargetModels import TargetModelMicroGIF
from plot import plot_spike_trains_side_by_side, plot_spike_train_projection, plot_neuron

# num_pops = 2
num_pops = 4
pop_size = 2
# pop_size = 2
# pop_size = 1

for random_seed in range(3, 4):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    pop_sizes, snn = TargetModelMicroGIF.get_low_dim_micro_GIF_transposed(random_seed=random_seed)
    snn = microGIF(snn.get_parameters(), snn.N)

    N = snn.N
    # pop_sizes = [8, 2, 9, 2]
    # neurons_coeff = torch.cat([T(pop_sizes[0] * [0.]), T(pop_sizes[1] * [0.]), T(pop_sizes[2] * [0.25]), T(pop_sizes[3] * [0.1])])
    # inputs = sine_modulated_white_noise(t=1200*4, N=snn.N, neurons_coeff=neurons_coeff)
    # inputs = strong_sine_modulated_white_noise(t=4800, N=snn.N, neurons_coeff=neurons_coeff)
    # inputs = sine_input(t=4800, N=snn.N, neurons_coeff=neurons_coeff)

    A_coeff_1 = torch.randn((4,))
    A_coeff_2 = torch.randn((4,))
    phase_shifts_1 = torch.rand((4,))
    phase_shifts_2 = phase_shifts_1 + torch.rand((4,))
    inputs_1 = experiments.white_noise_sum_of_sinusoids(t=1200, A_coeff=A_coeff_1, phase_shifts=phase_shifts_1)
    # inputs_2 = experiments.white_noise_sum_of_sinusoids(t=t, A_coeff=A_coeff_2, phase_shifts=phase_shifts_2)

    inputs = torch.vstack([inputs_1, torch.zeros_like(inputs_1)])
    for _ in range(N - 2):
        inputs = torch.vstack([inputs, torch.zeros_like(inputs_1)])
    inputs = inputs.T

    print('- SNN test for class {} -'.format(snn.__class__.__name__))
    print('#inputs sum: {}'.format(inputs.sum()))
    _, spikes, vs = model_util.feed_inputs_sequentially_return_args(snn, inputs)
    spikes = spikes.clone().detach()
    # spikes = model_util.feed_inputs_sequentially_return_spike_train(snn, inputs)
    # print('snn weights: {}'.format(snn.w))
    hard_thresh_spikes_sum = torch.round(spikes).sum()
    print('spikes sum: {}'.format(hard_thresh_spikes_sum))
    soft_thresh_spikes_sum = (spikes > 0.333).sum()
    zero_thresh_spikes_sum = (spikes > 0).sum()
    print('thresholded spikes sum: {}'.format(torch.round(spikes).sum()))
    print('=========avg. hard rate: {}'.format(1000*hard_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))
    print('=========avg. soft rate: {}'.format(1000*soft_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))
    print('=========avg. zero thresh rate: {}'.format(1000*zero_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))
    plot_spike_train_projection(spikes, fname='test_projection_{}_ext_input'.format(snn.__class__.__name__) + '_' + str(random_seed))

    # plot_neuron(membrane_potentials.detach().numpy(), title='{} neuron plot ({:.2f} spikes)'.format(snn.__class__.__name__, spikes.sum()),
    #             uuid='test', fname='test_{}_poisson_input'.format(snn.__class__.__name__) + '_' + str(random_seed))

    zeros = torch.zeros_like(inputs)
    _, membrane_potentials_zeros, spikes_zeros = model_util.feed_inputs_sequentially_return_args(snn, zeros)
    # spikes_zeros = model_util.feed_inputs_sequentially_return_spike_train(snn, zeros)
    # plot_neuron(membrane_potentials_zeros.detach().numpy(), title='Neuron plot ({:.2f} spikes)'.format(spikes_zeros.sum()),
                # uuid='test', fname='test_LIF_no_input'  + '_' + str(random_seed))
    print('#spikes no input: {}'.format(torch.round(spikes_zeros).sum()))

    # plot_spiketrains_side_by_side(spikes, spikes_zeros, 'test_LIF', title='Test LIF spiketrains random and zero input', legend=['Poisson input', 'No input'])
    snn.w = torch.nn.Parameter(torch.zeros((snn.v.shape[0],snn.v.shape[0])), requires_grad=True)
    # spikes_zero_weights = model_util.feed_inputs_sequentially_return_spike_train(snn, inputs)
    _, membrane_potentials_zero_weights, spikes_zero_weights = model_util.feed_inputs_sequentially_return_args(snn, inputs)
    print('#spikes no weights: {}'.format(torch.round(spikes_zero_weights).sum()))
    plot_neuron(inputs.detach().numpy(), title='I_ext'.format(snn.name(), spikes.sum()),
                uuid='test', fname='test_ext_input'.format(snn.name()) + '_' + str(random_seed))
    plot_neuron(vs.detach().numpy(), title='{} neuron plot ({:.2f} spikes)'.format(snn.name(), spikes.sum()),
                uuid='test', fname='test_membrane_potential_{}_ext_input'.format(snn.name()) + '_' + str(random_seed))
    plot_spike_trains_side_by_side(spikes, spikes_zero_weights, 'test_{}'.format(snn.__class__.__name__),
                                   title='Test {} spiketrains random input'.format(snn.__class__.__name__),
                                   legend=['Random weights', 'No weights'])
    plot_spike_trains_side_by_side(spikes, spikes_zeros, 'test_{}'.format(snn.__class__.__name__),
                                   title='Test {} spiketrains no input'.format(snn.__class__.__name__),
                                   legend=['Random weights', 'No input'])

    tau_vr = torch.tensor(5.0)
    loss = spike_metrics.van_rossum_dist(spikes, spikes_zeros, tau=tau_vr)
    print('tau_vr: {}, loss: {}'.format(tau_vr, loss))
    loss_rate = spike_metrics.firing_rate_distance(spikes, spikes_zero_weights)
    print('firing rate loss: {}'.format(loss_rate))


sys.exit(0)
