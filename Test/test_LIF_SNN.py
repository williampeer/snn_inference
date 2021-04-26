import torch

import model_util
import spike_metrics
from TargetModels.TargetEnsembleModels import lif_ensembles_model_dales_compliant, \
    lif_continuous_ensembles_model_dales_compliant
from experiments import poisson_input, draw_from_uniform, continuous_normalised_poisson_noise
from plot import plot_neuron, plot_spiketrains_side_by_side

num_neurons = 12

for random_seed in range(1, 4):
    # snn = lif_ensembles_model_dales_compliant(random_seed=random_seed)
    snn = lif_continuous_ensembles_model_dales_compliant(random_seed=random_seed)

    # inputs = poisson_input(10., t=4000, N=snn.N)  # now assumes rate in Hz
    inputs = continuous_normalised_poisson_noise(4., t=4000, N=snn.N)  # now assumes rate in Hz

    print('#inputs: {}'.format(inputs.sum()))
    membrane_potentials, spikes = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, inputs)
    print('#spikes: {}'.format(spikes.sum()))
    plot_neuron(membrane_potentials.detach().numpy(), title='LIF neuron plot ({:.2f} spikes)'.format(spikes.sum()), uuid='test', fname='test_LIF_poisson_input' + '_' + str(random_seed))

    zeros = torch.zeros_like(inputs)
    membrane_potentials_zeros, spikes_zeros = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, zeros)
    plot_neuron(membrane_potentials_zeros.detach().numpy(), title='Neuron plot ({:.2f} spikes)'.format(spikes_zeros.sum()), uuid='test', fname='test_LIF_no_input'  + '_' + str(random_seed))

    plot_spiketrains_side_by_side(spikes, spikes_zeros, 'test_LIF', title='Test LIF spiketrains random and zero input', legend=['Poisson input', 'No input'])

    tau_vr = torch.tensor(4.0)
    loss = spike_metrics.van_rossum_dist(spikes, spikes_zeros, tau=tau_vr)
    print('tau_vr: {}, loss: {}'.format(tau_vr, loss))
