import torch

import model_util
import spike_metrics
from TargetModels import TargetModels
from experiments import continuous_normalised_poisson_noise
from plot import plot_neuron, plot_spiketrains_side_by_side

num_neurons = 12

for random_seed in range(1, 6):
    # snn = lif_ensembles_model_dales_compliant(random_seed=random_seed)
    # snn = TargetModels.lif_continuous_ensembles_model_dales_compliant(random_seed=random_seed)
    # snn = TargetModels.lif_HS_17_continuous_ensembles_model_dales_compliant(random_seed=random_seed)
    # snn = TargetModels.lif_r_continuous_ensembles_model_dales_compliant(random_seed=random_seed)
    snn = TargetModels.lif_r_asc_continuous_ensembles_model_dales_compliant(random_seed=random_seed)
    # snn = TargetModels.glif_continuous_ensembles_model_dales_compliant(random_seed=random_seed)

    # inputs = poisson_input(10., t=4000, N=snn.N)  # now assumes rate in Hz
    inputs = continuous_normalised_poisson_noise(10., t=1000, N=snn.N)  # now assumes rate in Hz

    print('- SNN test for class {} -'.format(snn.__class__.__name__))
    print('#inputs: {}'.format(inputs.sum()))
    membrane_potentials, spikes = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, inputs)
    # print('snn weights: {}'.format(snn.w))
    print('#spikes: {}'.format(spikes.sum()))
    print('=========avg. rate: {}'.format(1000*torch.round(spikes).sum() / (spikes.shape[1] * spikes.shape[0])))
    plot_neuron(membrane_potentials.detach().numpy(), title='{} neuron plot ({:.2f} spikes)'.format(snn.__class__.__name__, spikes.sum()),
                uuid='test', fname='test_{}_poisson_input'.format(snn.__class__.__name__) + '_' + str(random_seed))

    zeros = torch.zeros_like(inputs)
    membrane_potentials_zeros, spikes_zeros = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, zeros)
    plot_neuron(membrane_potentials_zeros.detach().numpy(), title='Neuron plot ({:.2f} spikes)'.format(spikes_zeros.sum()),
                uuid='test', fname='test_LIF_no_input'  + '_' + str(random_seed))
    print('#spikes no input: {}'.format(spikes_zeros.sum()))

    # plot_spiketrains_side_by_side(spikes, spikes_zeros, 'test_LIF', title='Test LIF spiketrains random and zero input', legend=['Poisson input', 'No input'])
    snn.w = torch.nn.Parameter(torch.zeros((snn.v.shape[0],snn.v.shape[0])), requires_grad=True)
    membrane_potentials_zero_weights, spikes_zero_weights = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, inputs)
    print('#spikes no weights: {}'.format(spikes_zero_weights.sum()))
    # plot_neuron(membrane_potentials.detach().numpy(), title='LIF neuron plot ({:.2f} spikes)'.format(spikes.sum()),
    #             uuid='test', fname='test_LIF_poisson_input' + '_' + str(random_seed))
    plot_spiketrains_side_by_side(spikes, spikes_zero_weights, 'test_{}'.format(snn.__class__.__name__),
                                  title='Test {} spiketrains random input'.format(snn.__class__.__name__),
                                  legend=['Random weights', 'No weights'])

    tau_vr = torch.tensor(4.0)
    loss = spike_metrics.van_rossum_dist(spikes, spikes_zeros, tau=tau_vr)
    print('tau_vr: {}, loss: {}'.format(tau_vr, loss))
