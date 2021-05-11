import torch


# an approximation using torch.where
def torch_van_rossum_convolution(spikes, tau):
    decay_kernel = torch.exp(-torch.tensor(1.) / tau)
    convolved_spiketrain = spikes.clone()
    one_row_of_zeros = torch.zeros((1, spikes.shape[1]))
    for i in range(int(3*tau)):
        tmp_shifted_conv = torch.cat([one_row_of_zeros, convolved_spiketrain[:-1]])
        # sig(v - threshold) = 0.5 for v = threshold
        convolved_spiketrain = torch.where(spikes < 0.75, tmp_shifted_conv.clone() * decay_kernel, spikes.clone())
    return convolved_spiketrain


def torch_van_rossum_convolution_two_sided(spikes, tau):
    decay_kernel = torch.exp(-torch.tensor(1.) / tau)
    convolved_spiketrain = spikes.clone()
    # convolved_spiketrain_backwards = spikes.clone()
    row_of_zeros = torch.zeros((1, spikes.shape[1]))
    for i in range(int(3*tau)):
        tmp_shifted_conv = torch.cat([row_of_zeros, convolved_spiketrain[:-1]])
        tmp_shifted_backwards = torch.cat([convolved_spiketrain[1:], row_of_zeros.clone().detach()])
        # sig(v - threshold) = 0.5 for v = threshold
        convolved_spiketrain = torch.where(spikes < 0.75, torch.max(tmp_shifted_conv * decay_kernel, tmp_shifted_backwards * decay_kernel), spikes.clone())
        # convolved_spiketrain_backwards = torch.where(spikes < 0.75, tmp_shifted_backwards * decay_kernel, spikes.clone())
    return convolved_spiketrain


def van_rossum_dist(spikes, target_spikes, tau):
    c1 = torch_van_rossum_convolution(spikes=spikes, tau=tau)
    c2 = torch_van_rossum_convolution(spikes=target_spikes, tau=tau)
    return euclid_dist(c1, c2)


def van_rossum_dist_two_sided(spikes, target_spikes, tau):
    c1 = torch_van_rossum_convolution_two_sided(spikes=spikes, tau=tau)
    c2 = torch_van_rossum_convolution_two_sided(spikes=target_spikes, tau=tau)
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
    return torch.sqrt(torch.pow(torch.sub(spikes2, spikes1), 2).sum() + 1e-18)


def mse(s1, s2):
    return torch.pow(torch.sub(s2, s1), 2).sum() / (s1.shape[0] * s1.shape[1])


def van_rossum_squared_distance(s1, s2, tau):
    c1 = torch_van_rossum_convolution(spikes=s1, tau=tau)
    c2 = torch_van_rossum_convolution(spikes=s2, tau=tau)
    return mse(c1, c2)


def silent_penalty_term(spikes, targets):
    normalised_spike_rate = spikes.sum(dim=0) / (spikes.shape[0] * spikes.shape[1])
    normalised_target_rate = targets.sum(dim=0) / (targets.shape[0] * targets.shape[1])
    # f_penalty(x,y) = se^(-x/(N*t))
    return torch.pow(torch.exp(-normalised_spike_rate) - torch.exp(-normalised_target_rate), 2).sum()
    # return torch.exp(-normalised_spike_rate).sum()

# def silent_penalty_term(model_spikes, target_spikes):
#     mean_model_rate = model_spikes.sum(dim=0)
#     mean_targets_rate = target_spikes.sum(dim=0)
#     # assert model_spikes.shape[0] > model_spikes.shape[1]
#     T = model_spikes.shape[0] / 1000.
#     # f_penalty(x,y) = sqrt(pow(e^(-x/T.) - e^(-y/T.)).sum() + 1e-18)
#     silent_penalty = torch.sqrt(torch.pow(torch.exp(-mean_targets_rate/torch.tensor(T)) - torch.exp(-mean_model_rate/torch.tensor(T)), 2).sum()+1e-18) / model_spikes.shape[1]
#     return silent_penalty


def firing_rate_distance(model_spikes, target_spikes):
    mean_model_rate = model_spikes.sum(dim=0)
    mean_targets_rate = target_spikes.sum(dim=0)
    return euclid_dist(mean_targets_rate, mean_model_rate) / (model_spikes.shape[0])
    # assert model_spikes.shape[0] > model_spikes.shape[1]
    # f_penalty(x,y) = sqrt(pow(e^(-x/T.) - e^(-y/T.)).sum() + 1e-18)
    # silent_penalty = torch.sqrt(torch.pow(torch.exp(-mean_model_rate/torch.tensor(T)) - torch.exp(-mean_targets_rate/torch.tensor(T)), 2).sum()+1e-18) / model_spikes.shape[1]
    # return torch.sub(mean_targets_rate, mean_model_rate).sum() / (model_spikes.shape[0] * model_spikes.shape[1])
    # return torch.sqrt(torch.pow(torch.sub(mean_targets_rate, mean_model_rate), 2).sum() + 1e-18)


def normalised_overall_activity_term(model_spikes):
    # overall-activity penalty:
    return (model_spikes.sum() + 1e-09) / model_spikes.shape[1]


def shortest_dist_rates(spikes, target_spikes):
    assert spikes.shape[0] > spikes.shape[1], "each time step as a row expected, meaning column by node"

    spike_rates = spikes.sum(dim=0) * 1000. / spikes.shape[0]
    spike_rates, _ = torch.sort(spike_rates)
    target_rates = target_spikes.sum(axis=0) * 1000. / target_spikes.shape[0]
    target_rates, _ = torch.sort(target_rates)

    return torch.sqrt(torch.pow(torch.sub(spike_rates, target_rates), 2).sum() + 1e-18)


def shortest_dist_rates_w_silent_penalty(spikes, target_spikes):
    assert spikes.shape[0] > spikes.shape[1], "each time step as a row expected, meaning column by node"

    spike_rates = spikes.sum(dim=0) * 1000. / spikes.shape[0]
    spike_rates, _ = torch.sort(spike_rates)
    target_rates = target_spikes.sum(dim=0) * 1000. / target_spikes.shape[0]
    target_rates, _ = torch.sort(target_rates)

    silent_penalty = torch.sqrt(torch.pow(torch.exp(-spike_rates) - torch.exp(target_rates), 2).sum() + 1e-18)
    return torch.sqrt(torch.pow(torch.sub(spike_rates, target_rates), 2).sum() + 1e-18) + silent_penalty

