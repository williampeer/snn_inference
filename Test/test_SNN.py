import sys

import numpy as np
import torch

import model_util
import spike_metrics
from TargetModels import TargetModels
from experiments import sine_modulated_white_noise
from plot import plot_spike_trains_side_by_side, plot_spike_train_projection

num_neurons = 10

for random_seed in range(3, 7):
    # snn = lif_ensembles_model_dales_compliant(random_seed=random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    # model_class = LIF_soft_weights_only
    # init_params_model = draw_from_uniform(model_class.parameter_init_intervals, num_neurons)
    # snn = model_class(init_params_model)
    # snn = TargetModels.lif_continuous_ensembles_model_dales_compliant(random_seed=random_seed, N=num_neurons)
    # snn = TargetModels.lif_r_continuous_ensembles_model_dales_compliant(random_seed=random_seed, N=num_neurons)
    # snn = TargetModels.lif_r_asc_continuous_ensembles_model_dales_compliant(random_seed=random_seed, N=num_neurons)
    snn = TargetModels.glif_continuous_ensembles_model_dales_compliant(random_seed=random_seed, N=num_neurons)

    # inputs = poisson_input(10., t=4000, N=snn.N)  # now assumes rate in Hz
    inputs = sine_modulated_white_noise(10., t=12000, N=snn.N)  # now assumes rate in Hz

    print('- SNN test for class {} -'.format(snn.__class__.__name__))
    print('#inputs: {}'.format(inputs.sum()))
    # membrane_potentials, spikes = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, inputs)
    spikes = model_util.feed_inputs_sequentially_return_spike_train(snn, inputs)
    # print('snn weights: {}'.format(snn.w))
    hard_thresh_spikes_sum = torch.round(spikes).sum()
    print('spikes sum: {}'.format(hard_thresh_spikes_sum))
    soft_thresh_spikes_sum = (spikes > 0.333).sum()
    zero_thresh_spikes_sum = (spikes > 0).sum()
    print('thresholded spikes sum: {}'.format(torch.round(spikes).sum()))
    print('=========avg. hard rate: {}'.format(1000*hard_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))
    print('=========avg. soft rate: {}'.format(1000*soft_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))
    print('=========avg. zero thresh rate: {}'.format(1000*zero_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))
    plot_spike_train_projection(spikes, fname='test_projection_{}_poisson_input'.format(snn.__class__.__name__) + '_' + str(random_seed))

    # plot_neuron(membrane_potentials.detach().numpy(), title='{} neuron plot ({:.2f} spikes)'.format(snn.__class__.__name__, spikes.sum()),
    #             uuid='test', fname='test_{}_poisson_input'.format(snn.__class__.__name__) + '_' + str(random_seed))

    zeros = torch.zeros_like(inputs)
    # membrane_potentials_zeros, spikes_zeros = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, zeros)
    spikes_zeros = model_util.feed_inputs_sequentially_return_spike_train(snn, zeros)
    # plot_neuron(membrane_potentials_zeros.detach().numpy(), title='Neuron plot ({:.2f} spikes)'.format(spikes_zeros.sum()),
                # uuid='test', fname='test_LIF_no_input'  + '_' + str(random_seed))
    print('#spikes no input: {}'.format(torch.round(spikes_zeros).sum()))

    # plot_spiketrains_side_by_side(spikes, spikes_zeros, 'test_LIF', title='Test LIF spiketrains random and zero input', legend=['Poisson input', 'No input'])
    snn.w = torch.nn.Parameter(torch.zeros((snn.v.shape[0],snn.v.shape[0])), requires_grad=True)
    # membrane_potentials_zero_weights, spikes_zero_weights = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, inputs)
    spikes_zero_weights = model_util.feed_inputs_sequentially_return_spike_train(snn, inputs)
    print('#spikes no weights: {}'.format(torch.round(spikes_zero_weights).sum()))
    # plot_neuron(membrane_potentials.detach().numpy(), title='LIF neuron plot ({:.2f} spikes)'.format(spikes.sum()),
    #             uuid='test', fname='test_LIF_poisson_input' + '_' + str(random_seed))
    plot_spike_trains_side_by_side(spikes, spikes_zero_weights, 'test_{}'.format(snn.__class__.__name__),
                                   title='Test {} spiketrains random input'.format(snn.__class__.__name__),
                                   legend=['Random weights', 'No weights'])

    tau_vr = torch.tensor(5.0)
    loss = spike_metrics.van_rossum_dist(spikes, spikes_zeros, tau=tau_vr)
    print('tau_vr: {}, loss: {}'.format(tau_vr, loss))
    loss_rate = spike_metrics.firing_rate_distance(spikes, spikes_zero_weights)
    print('firing rate loss: {}'.format(loss_rate))


sys.exit(0)
