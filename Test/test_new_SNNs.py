import torch
import numpy as np

import model_util
import spike_metrics
from TargetModels import TargetEnsembleModels
from experiments import poisson_input
from plot import plot_spiketrains_side_by_side

for random_seed in range(5, 10):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    num_neurons = 12

    # snn = TargetEnsembleModels.lif_r_ensembles_model_dales_compliant(random_seed=random_seed, N = 12)
    # ext_name = 'ensembles_{}_dales_LIF_R'.format(random_seed)
    # snn = TargetEnsembleModels.lif_asc_ensembles_model_dales_compliant(random_seed=random_seed, N=12)
    # ext_name = 'ensembles_{}_dales_LIF_ASC'.format(random_seed)
    snn = TargetEnsembleModels.lif_r_asc_ensembles_model_dales_compliant(random_seed=random_seed, N=12)
    ext_name = 'ensembles_{}_dales_LIF_R_ASC'.format(random_seed)

    # Izhikevich?
    # snn = TargetEnsembleModels.izhikevich_ensembles_model_dales_compliant(random_seed=random_seed, N=12)
    # ext_name = 'ensembles_{}_dales_Izhikevich'.format(random_seed)

    rate = 10.
    inputs = poisson_input(rate, t=12000, N=snn.N)  # rate in Hz
    print('#inputs: {}'.format(inputs.sum()))
    # membrane_potentials, spikes = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, inputs)
    spikes = model_util.feed_inputs_sequentially_return_spiketrain(snn, inputs)
    print('#spikes: {}'.format(torch.round(spikes).sum(dim=0)))
    # plot_neuron(membrane_potentials.data, title='GLIF neuron plot ({:.2f} spikes)'.format(spikes.sum()), fname_ext='test_GLIF_poisson_input' + '_' + str(random_seed))
    assert spikes.sum() < 12. * 12 * 12 * 2, "should be less spikes than input rate * 2 per neuron. spikes.sum(): {}".format(spikes.sum())

    zeros = torch.zeros_like(inputs)
    # membrane_potentials_zeros, spikes_zeros = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, zeros)
    spikes_zeros = model_util.feed_inputs_sequentially_return_spiketrain(snn, zeros)
    # plot_neuron(membrane_potentials_zeros.data, title='Neuron plot ({:.2f} spikes)'.format(spikes_zeros.sum()), fname_ext='test_GLIF_no_input'  + '_' + str(random_seed))

    plot_spiketrains_side_by_side(spikes, spikes_zeros, 'test_SNNs', title='{} random ({} Hz) and zero input'.format(ext_name, rate),
                                  legend=['Poisson input', 'No input'])

    tau_vr = torch.tensor(4.0)
    loss = spike_metrics.van_rossum_dist(spikes, spikes_zeros, tau=tau_vr)
    print('tau_vr: {}, loss: {}'.format(tau_vr, loss))
    assert loss > 1e-03, "should have loss for input vs. no input"
