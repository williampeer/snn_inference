# import torch
#
# import model_util
# import spike_metrics
# from Models.LIF import LIF
# from experiments import poisson_input, zip_dicts, randomise_parameters
# from plot import plot_neuron, plot_spiketrains_side_by_side
#
# static_parameters = {'N': 3}
# free_parameters = {'w_mean': 0.2, 'w_var': 0.3, 'tau_m': 1.5, 'tau_g': 4.0, 'v_rest': -60.0}
#
# snn = LIF(device='cpu', parameters=zip_dicts(static_parameters, free_parameters))
#
# inputs = poisson_input(0.5, t=500, N=static_parameters['N'])
# # inputs = torch.cat((torch.tensor([12*[4.]]), torch.zeros((100, static_parameters['N']))))
# # inputs = 4. * torch.ones((500, static_parameters['N']))
# membrane_potentials, spikes = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, inputs)
# plot_neuron(membrane_potentials.data, title='LIF neuron plot ({:.2f} spikes)'.format(spikes.sum()), fname_ext='test_LIF_poisson_input')
#
# zeros = torch.zeros_like(inputs)
# membrane_potentials_zeros, spikes_zeros = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, zeros)
# plot_neuron(membrane_potentials_zeros.data, title='LIF neuron plot ({:.2f} spikes)'.format(spikes_zeros.sum()), fname_ext='test_LIF_no_input')
#
# plot_spiketrains_side_by_side(spikes, spikes_zeros, 'test_LIF', title='Test LIF spiketrains random and zero input')
#
# tau_vr = torch.tensor(2.0)
# loss = spike_metrics.van_rossum_dist(spikes, spikes_zeros, tau=tau_vr)
# print('tau_vr: {}, loss: {}'.format(tau_vr, loss))


import torch

import model_util
import spike_metrics
from Models.LIF import LIF
from experiments import poisson_input, draw_from_uniform
from plot import plot_neuron, plot_spiketrains_side_by_side

num_neurons = 12

for random_seed in range(1, 6):
    # static_parameters = {'N': 10}
    # free_parameters = {'w_mean': 0.2, 'w_var': 0.3}
    # snn = GLIF(device='cpu', parameters=zip_dicts(static_parameters, free_parameters))
    init_params_model = draw_from_uniform(LIF.parameter_init_intervals, num_neurons)
    snn = LIF(parameters=init_params_model)
    # snn = TargetModels.glif1(N = 12); ext_name = '1'
    # snn = TargetModels.glif1_2(N = 12); ext_name = '1_2'
    # snn = TargetModels.glif2(N = 12); ext_name = '2'
    # snn = TargetModels.glif3(N = 12); ext_name = '3'
    # snn = TargetModels.glif_async(N = 12); ext_name = 'glif_async'
    # snn = TargetModels.glif_slower_more_synchronous(N = 12); ext_name = 'glif_slower_more_synchronous'

    inputs = poisson_input(5., t=4000, N=snn.N)  # now assumes rate in Hz
    print('#inputs: {}'.format(inputs.sum()))
    membrane_potentials, spikes = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, inputs)
    print('#spikes: {}'.format(spikes.sum()))
    plot_neuron(membrane_potentials.data, title='LIF neuron plot ({:.2f} spikes)'.format(spikes.sum()), fname_ext='test_LIF_poisson_input' + '_' + str(random_seed))

    zeros = torch.zeros_like(inputs)
    membrane_potentials_zeros, spikes_zeros = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, zeros)
    plot_neuron(membrane_potentials_zeros.data, title='Neuron plot ({:.2f} spikes)'.format(spikes_zeros.sum()), fname_ext='test_LIF_no_input'  + '_' + str(random_seed))

    plot_spiketrains_side_by_side(spikes, spikes_zeros, 'test_LIF', title='Test LIF spiketrains random and zero input', legend=['Poisson input', 'No input'])

    tau_vr = torch.tensor(4.0)
    loss = spike_metrics.van_rossum_dist(spikes, spikes_zeros, tau=tau_vr)
    print('tau_vr: {}, loss: {}'.format(tau_vr, loss))
