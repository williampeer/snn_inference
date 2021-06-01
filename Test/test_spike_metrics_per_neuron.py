from experiments import poisson_input

import torch

from stats import firing_rate_per_neuron


def test_sum_per_node():
    t = 400; N = 12; tau_vr = torch.tensor(20.0)

    # zeros = torch.zeros((t, N))
    s1 = (poisson_input(0.8, t, N) > 0).float()
    s2 = (poisson_input(0.8, t, N) > 0).float()
    # mse_nodes = mse_per_node(s1, s2)
    # assert len(mse_nodes) == s1.shape[1], "mse should be per column (node). mse shape: {}, s1.shape: {}"\
    #     .format(mse_nodes.shape, s1.shape)

    # vr_dist_nodes = van_rossum_dist_per_node(s1, s2, tau_vr)
    # assert len(vr_dist_nodes) == s1.shape[1], "vr_dist_nodes should be per column (node). vr_dist_nodes shape: {}, s1.shape: {}"\
    #     .format(vr_dist_nodes.shape, s1.shape)

    # vr_squared_nodes = van_rossum_squared_per_node(s1, s2, tau_vr)
    # assert len(vr_squared_nodes) == s1.shape[1], "vr_squared_nodes should be per column (node). vr_squared_nodes shape: {}, s1.shape: {}" \
    #     .format(vr_squared_nodes.shape, s1.shape)


def test_firing_rate_per_neuron():
    N = 12; t=10000
    rate = 10.
    spikes = (poisson_input(rate=rate, t=t, N=N) > 0).float()
    rates = firing_rate_per_neuron(spikes) * t/1000.
    assert rates.shape[0] == N, "rates should be per node"
    # broadcast_rates = rate * torch.ones_like(rates)
    # assert torch.all((broadcast_rates - 0.05 * broadcast_rates < rates < broadcast_rates + broadcast_rates * 0.05).bool()), \
        # "broadcast_rates: {} should be approx. rates: {}".format(broadcast_rates, rates)
    assert torch.all((rate - 0.05 * rate < rates).bool()), \
        "rate: {} should be approx. rates: {}".format(rate, rates)
    assert torch.all((rates < rate + rate * 0.05).bool()), \
        "rate: {} should be approx. rates: {}".format(rate, rates)

    print('# spikes: {}'.format(spikes.sum()))
    print('mean rates: {}'.format(rates))


# test_sum_per_node()
test_firing_rate_per_neuron()
