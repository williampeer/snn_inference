import numpy as np
import torch


def mean_firing_rate(spikes, bin_size=1.):
    assert spikes.shape[0] > spikes.shape[1], "should be bins (1ms) by nodes (rows by cols)"
    return spikes.sum(axis=0) / (spikes.shape[0] * bin_size)


def sums_helper(spikes1, spikes2):
    assert spikes1.shape[0] > spikes1.shape[1], "expected one column per neuron. spikes1.shape: {}".format(spikes1.shape)
    # sum over bins
    sum_spikes1 = spikes1.sum(axis=1)
    sum_spikes2 = spikes2.sum(axis=1)
    return torch.reshape(torch.cat([sum_spikes1, sum_spikes2]), (2, -1))  # N by spikes


def firing_rate_per_neuron(spikes):
    assert spikes.shape[0] > spikes.shape[1], "should be bins (1ms) by nodes (rows by cols)"
    return torch.mean(spikes, dim=0)


def binned_avg_firing_rate_per_neuron(spikes, bin_size):
    spikes = np.array(spikes)
    assert spikes.shape[0] > spikes.shape[1], "should be bins (1ms) by nodes (rows by cols)"

    std_per_node = torch.zeros((spikes.shape[1],))
    mean_per_node = torch.zeros((spikes.shape[1],))
    for node_i in range(spikes.shape[1]):
        avgs = binned_firing_rates(spikes[:, node_i], bin_size)
        std_per_node[node_i], mean_per_node[node_i] = torch.std_mean(torch.tensor(avgs))
    return std_per_node, mean_per_node


def binned_firing_rates(vec, bin_size):
    num_intervals = int(vec.shape[0] / bin_size)
    avgs = np.zeros((num_intervals,))

    for i in range(num_intervals):
        cur_interval = vec[i*bin_size:(i+1)*bin_size]
        avgs[i] = np.mean(cur_interval)

    return avgs


def pairwise_correlation(n1, n2, bin_size):
    binned_rates1 = binned_firing_rates(n1, bin_size)
    binned_rates2 = binned_firing_rates(n2, bin_size)
    binned_rates = np.array([binned_rates1, binned_rates2])

    var1 = np.var(binned_rates1)
    var2 = np.var(binned_rates2)

    rates_cov = np.cov(binned_rates)
    rho = rates_cov[0][1] / np.sqrt(var1 * var2)
    return rho


def spike_train_correlation(s1, s2, bin_size=100):
    s1 = np.array(s1); s2 = np.array(s2)
    assert s1.shape[0] == s2.shape[0] and s1.shape[1] == s2.shape[1], "shapes should be equal. s1.shape: {}, s2.shape: {}".format(s1.shape, s2.shape)

    pairwise_correlations = np.zeros((s1.shape[1], s2.shape[1]))

    for node_i in range(s1.shape[1]):
        for node_j in range(s2.shape[1]):
            pairwise_correlations[node_i, node_j] = pairwise_correlation(s1[:, node_i], s2[:, node_j], bin_size)
    return pairwise_correlations
