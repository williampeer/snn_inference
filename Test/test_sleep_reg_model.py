import torch

import model_util
import spike_metrics
from Models.SleepRegulationModel import SleepRegulationModel
from experiments import poisson_input
from plot import plot_neuron, plot_spiketrains_side_by_side

snn = SleepRegulationModel(1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

sample_inputs = poisson_input(0.6, t=400, N=3)
membrane_potentials, spikes = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, sample_inputs)
plot_neuron(membrane_potentials.data, title='Neuron plot ({:.2f} spikes)'.format(spikes.sum()), fname_ext='test_sleep_reg_model_poisson_input')

zeros = torch.zeros_like(sample_inputs)
membrane_potentials_zeros, spikes_zeros = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, zeros)
plot_neuron(membrane_potentials_zeros.data, title='Neuron plot ({:.2f} spikes)'.format(spikes_zeros.sum()), fname_ext='test_sleep_reg_model_no_input')

plot_spiketrains_side_by_side(spikes, spikes_zeros, 'test_Izhikevich', title='Spiketrains random and zero input (Sleep Reg. Model)')

tau_vr = torch.tensor(20.0)
loss_vs_zeros = spike_metrics.van_rossum_dist(spikes, spikes_zeros, tau=tau_vr)
print('tau_vr: {}, loss: {}'.format(tau_vr, loss_vs_zeros))
assert spikes.sum() > 0, "model was silent, i.e. no spikes"
sample_targets = poisson_input(0.5, t=400, N=3)
loss_vs_sample_targets = spike_metrics.van_rossum_dist(spikes, sample_targets, tau_vr)
assert loss_vs_sample_targets > loss_vs_zeros, "loss_vs_sample_targets: {} should be less versus spikes than versus loss_vs_zeros: {}.".format(loss_vs_sample_targets, loss_vs_zeros)
