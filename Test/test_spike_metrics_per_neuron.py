from experiments import poisson_input

import torch

from spike_metrics import mse_per_node, van_rossum_dist_per_node, van_rossum_squared_per_node, firing_rate_per_neuron


def test_sum_per_node():
    t = 400; N = 12; tau_vr = torch.tensor(20.0)

    # zeros = torch.zeros((t, N))
    s1 = (poisson_input(0.8, t, N) > 0).float()
    s2 = (poisson_input(0.8, t, N) > 0).float()
    mse_nodes = mse_per_node(s1, s2)
    assert len(mse_nodes) == s1.shape[1], "mse should be per column (node). mse shape: {}, s1.shape: {}"\
        .format(mse_nodes.shape, s1.shape)

    vr_dist_nodes = van_rossum_dist_per_node(s1, s2, tau_vr)
    assert len(vr_dist_nodes) == s1.shape[1], "vr_dist_nodes should be per column (node). vr_dist_nodes shape: {}, s1.shape: {}"\
        .format(vr_dist_nodes.shape, s1.shape)

    vr_squared_nodes = van_rossum_squared_per_node(s1, s2, tau_vr)
    assert len(vr_squared_nodes) == s1.shape[1], "vr_squared_nodes should be per column (node). vr_squared_nodes shape: {}, s1.shape: {}" \
        .format(vr_squared_nodes.shape, s1.shape)


def test_firing_rate_per_neuron():
    N = 12; t=1000
    spikes = (poisson_input(0.75, t=t, N=N) > 0).float()
    rates = firing_rate_per_neuron(spikes)
    assert rates.shape[0] == N, "rates should be per node"
    rates_mean = torch.mean(rates)
    assert rates_mean - rates_mean * 0.05 < spikes.sum() / (N * t) < rates_mean + rates_mean * 0.05
    print('# spikes: {}'.format(spikes.sum()))
    print('mean rates: {}'.format(rates))


test_sum_per_node()
test_firing_rate_per_neuron()
