import torch

import model_util
import spike_metrics
from Models.Izhikevich import Izhikevich
from plot import plot_neuron

neuron = Izhikevich(device='cpu', parameters={'N': 1}, N=1)

ones = torch.ones((1000, 1))
membrane_potentials, model_spikes = model_util.feed_inputs_sequentially_return_tuple(neuron, ones)
plot_neuron(membrane_potentials.data, title="Izhikevich neuron test (1)", fname_ext='test_Izhikevich_neuron_1')
model_spikes_shifted = torch.cat([model_spikes.clone()[1:], torch.tensor([[0.]])])

cur_tau = torch.tensor(5.0)
loss = spike_metrics.van_rossum_dist(model_spikes, target_spikes=model_spikes_shifted, tau=cur_tau)
# plot_neuron(spike_metrics.torch_van_rossum_convolution(model_spikes, tau=cur_tau).data, title="Model spikes convolved")
# plot_neuron(spike_metrics.torch_van_rossum_convolution(model_spikes_shifted, tau=cur_tau).data,
#             title="Shifted model spikes convolved")
assert loss > 1e-09, "shifted model spikes should have loss"
print('shifted model spikes loss: {}'.format(loss))

rand_ones = 1.0 * (torch.rand((500, 10)) > 0.5)
loss_identical_trains = spike_metrics.van_rossum_dist(rand_ones, rand_ones, tau=cur_tau)
assert loss_identical_trains == 1e-09, "identical trains should have zero distance."
print('loss_identical_trains: {}'.format(loss_identical_trains))


zeros = torch.zeros_like(ones)
membrane_potentials, model_spikes = model_util.feed_inputs_sequentially_return_tuple(neuron, zeros)
plot_neuron(membrane_potentials.data, title="Izhikevich neuron test (2)", fname_ext='test_Izhikevich_neuron_2')
# transformed_potentials = torch.sigmoid(membrane_potentials)
loss = spike_metrics.van_rossum_dist(spikes=torch.zeros_like(membrane_potentials),
                                     target_spikes=torch.zeros_like(membrane_potentials), tau=cur_tau)
print('loss: {}'.format(loss))
assert loss == 1e-09, "zero loss should be 1e-09. loss was: {}".format(loss)
