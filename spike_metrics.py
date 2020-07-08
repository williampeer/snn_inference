import torch


# an approximation using torch.where
def torch_van_rossum_convolution(spikes, tau):
    decay_kernel = torch.exp(-torch.tensor(1.) / tau)
    convolved_spiketrain = spikes.clone()
    padding_zeros = torch.zeros((1, spikes.shape[1]))
    for i in range(int(3*tau)):
        tmp_shifted_conv = torch.cat([padding_zeros, convolved_spiketrain[:-1]])
        convolved_spiketrain = torch.where(spikes < 0.99, tmp_shifted_conv.clone() * decay_kernel, spikes.clone())
    return convolved_spiketrain


def van_rossum_dist(spikes, target_spikes, tau):
    c1 = torch_van_rossum_convolution(spikes=spikes, tau=tau)
    c2 = torch_van_rossum_convolution(spikes=target_spikes, tau=tau)
    return euclid_dist(c1, c2)


def euclid_dist(spikes1, spikes2):
    # sqrt((s1 - s2) ** 2)
    return torch.sqrt(torch.pow(torch.sub(spikes1, spikes2), 2).sum() + 1e-18)  # avoid sqrt(0) -> NaN


def mse(s1, s2):
    return torch.pow(torch.sub(s1, s2), 2).sum()


def van_rossum_squared_distance(s1, s2, tau):
    c1 = torch_van_rossum_convolution(spikes=s1, tau=tau)
    c2 = torch_van_rossum_convolution(spikes=s2, tau=tau)
    return mse(c1, c2)


# ------------- per neuron ---------------
def mse_per_node(s1, s2):
    return torch.pow(torch.sub(s1, s2), 2).sum(dim=0)


def euclid_dist_per_node(s1, s2):
    return torch.sqrt(torch.pow(torch.sub(s1, s2), 2).sum(dim=0) + 1e-18)  # avoid sqrt(0) -> NaN


def van_rossum_dist_per_node(s1, s2, tau):
    c1 = torch_van_rossum_convolution(spikes=s1, tau=tau)
    c2 = torch_van_rossum_convolution(spikes=s2, tau=tau)
    return euclid_dist_per_node(c1, c2)


def van_rossum_squared_per_node(s1, s2, tau):
    c1 = torch_van_rossum_convolution(spikes=s1, tau=tau)
    c2 = torch_van_rossum_convolution(spikes=s2, tau=tau)
    return mse_per_node(c1, c2)
