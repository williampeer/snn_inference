import torch
import numpy as np


# an approximation using torch.where
from stats import mean_firing_rate


def torch_van_rossum_convolution(spikes, tau):
    decay_kernel = torch.exp(-torch.tensor(1.) / tau)
    convolved_spiketrain = spikes.clone()
    padding_zeros = torch.zeros((1, spikes.shape[1]))
    for i in range(int(3*tau)):
        tmp_shifted_conv = torch.cat([padding_zeros, convolved_spiketrain[:-1]])
        # sig(v - threshold) = 0.5 for v = threshold
        convolved_spiketrain = torch.where(spikes < 0.5, tmp_shifted_conv.clone() * decay_kernel, spikes.clone())
    return convolved_spiketrain


def van_rossum_dist(spikes, target_spikes, tau):
    c1 = torch_van_rossum_convolution(spikes=spikes, tau=tau)
    c2 = torch_van_rossum_convolution(spikes=target_spikes, tau=tau)
    return euclid_dist(c1, c2)


def van_rossum_dist_one_to_K(spikes, target_spikes, tau):
    c1 = torch_van_rossum_convolution(spikes=spikes.reshape((-1, 1)), tau=tau)
    c1 = torch.ones((1, target_spikes.shape[1])) * c1.reshape((-1, 1))  # broadcast
    c2 = torch_van_rossum_convolution(spikes=target_spikes, tau=tau)
    euclid_per_node = torch.sqrt(torch.pow(torch.sub(c1, c2), 2) + 1e-18).sum(dim=0)
    return euclid_per_node


def greedy_shortest_dist_vr(spikes, target_spikes, tau):
    assert spikes.shape[0] > spikes.shape[1], "each time step as a row expected, meaning column by node"
    num_nodes = spikes.shape[1]
    indices_left = torch.arange(0, num_nodes)
    min_distances = torch.zeros((spikes.shape[1],))
    for s_i in range(0, num_nodes):
        d_i_J = van_rossum_dist_one_to_K(spikes[:, s_i], target_spikes[:, indices_left], tau)
        min_i_J = d_i_J[0]; min_index = 0
        for ind in range(1, d_i_J.shape[0]):
            if d_i_J[ind] < min_i_J:
                min_i_J = d_i_J[ind]
                min_index = ind
        min_distances[s_i] = min_i_J
        indices_left = indices_left[indices_left != min_index]

    return torch.mean(min_distances)


def euclid_dist(spikes1, spikes2):
    # sqrt((s1 - s2) ** 2)
    return torch.sqrt(torch.pow(torch.sub(spikes1, spikes2), 2).sum() + 1e-18) / (spikes1.shape[1])


def mse(s1, s2):
    return torch.pow(torch.sub(s1, s2), 2).sum()  / (s1.shape[0] * s1.shape[1])


def van_rossum_squared_distance(s1, s2, tau):
    c1 = torch_van_rossum_convolution(spikes=s1, tau=tau)
    c2 = torch_van_rossum_convolution(spikes=s2, tau=tau)
    return mse(c1, c2)


def firing_rate_distance(model_spikes, target_spikes):
    mean_model_rate = model_spikes.sum(axis=0)
    mean_targets_rate = target_spikes.sum(axis=0)
    # assert model_spikes.shape[0] > model_spikes.shape[1]
    T = model_spikes.shape[0] / 1000.
    # f_penalty(x,y) = sqrt(pow(e^(-x/T.) - e^(-y/T.)).sum() + 1e-18)
    silent_penalty = torch.sqrt(torch.pow(torch.exp(-mean_model_rate/torch.tensor(T)) - torch.exp(-mean_targets_rate/torch.tensor(T)), 2).sum()+1e-18) / model_spikes.shape[1]
    return torch.sqrt(torch.pow(torch.sub(mean_model_rate, mean_targets_rate), 2).sum() + 1e-18)/(model_spikes.shape[0]) \
           + silent_penalty
