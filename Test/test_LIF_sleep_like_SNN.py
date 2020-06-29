import torch
from torch import tensor as T
import model_util
from Models.LIF import LIF_sleep_model
from experiments import poisson_input
from plot import plot_neuron, plot_all_spiketrains, plot_spiketrain
from stats import firing_rate_per_neuron

snn = LIF_sleep_model()

inputs = poisson_input(0.5, t=4000, N=snn.N)
membrane_potentials, spikes = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, inputs)
mean_rates = firing_rate_per_neuron(spikes)
print('mean_rates (Poisson): {}'.format(mean_rates))
plot_neuron(membrane_potentials.data, title='Neuron plot ({:.2f} spikes) Poisson stim.'.format(spikes.sum()), fname_ext='test_LIF_complex_poisson_input')
snn.reset_hidden_state()

wake_mask = torch.cat([T(4*[1.]), T(4*[0.]), T(4*[0.])])
membrane_potentials, spikes_wake = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, inputs*wake_mask)
# plot_neuron(membrane_potentials.data, title='Neuron plot ({:.2f} spikes) wake stim.'.format(spikes_wake.sum()), fname_ext='test_LIF_complex_poisson_wake_input')
mean_rates = firing_rate_per_neuron(spikes_wake)
print('mean_rates (wake): {}'.format(mean_rates))
snn.reset_hidden_state()

rem_mask = torch.cat([T(4*[0.]), T(4*[1.0]), T(4*[0.])])
membrane_potentials, spikes_rem = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, inputs*rem_mask)
# plot_neuron(membrane_potentials.data, title='Neuron plot ({:.2f} spikes) REM stim.'.format(spikes_rem.sum()), fname_ext='test_LIF_complex_poisson_rem_input')
mean_rates = firing_rate_per_neuron(spikes_rem)
print('mean_rates (rem): {}'.format(mean_rates))
snn.reset_hidden_state()

nrem_mask = torch.cat([T(4*[0.]), T(4*[0.]), T(4*[1.0])])
membrane_potentials, spikes_nrem = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, inputs*nrem_mask)
# plot_neuron(membrane_potentials.data, title='Neuron plot ({:.2f} spikes) NREM stim.'.format(spikes_nrem.sum()), fname_ext='test_LIF_complex_poisson_nrem_input')
mean_rates = firing_rate_per_neuron(spikes_nrem)
print('mean_rates (nrem): {}'.format(mean_rates))
snn.reset_hidden_state()

# zeros = torch.zeros((1000, snn.N))
# membrane_potentials_zeros, spikes_zeros = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, zeros)
# plot_neuron(membrane_potentials_zeros.data, title='Neuron plot ({:.2f} spikes) no stimulus'.format(spikes_zeros.sum()), fname_ext='test_LIF_complex_no_input')
# mean_rates_zeros = firing_rate_per_neuron(spikes_zeros)
# print('mean_rates_zeros: {}'.format(mean_rates_zeros))

plot_spiketrain(spikes, title='Test LIF spiketrain Poisson input', uuid='test_LIF_complex')
# plot_spiketrains_side_by_side(spikes_rem, spikes_nrem, 'test_LIF_complex', title='Test LIF spiketrains REM & NREM input', legend=['REM', 'NREM'])
plot_all_spiketrains([spikes, spikes_wake, spikes_rem, spikes_nrem], 'test_LIF_complex', title='Test LIF spiketrains', legend=['Poisson', 'Wake', 'REM', 'NREM'])
