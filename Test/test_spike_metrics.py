from experiments import poisson_input
from spike_metrics import *

import torch


def test_mean_firing_rate():
    # firing rate
    a = 0.4 * torch.ones((10, 2))
    b = 0.6 * torch.ones((10, 2))
    tar_rate = 0.5
    sut = torch.reshape(torch.cat([a, b]), (-1, a.shape[1]))
    # check stack OK
    assert sut.shape[0] == (a.shape[0] + b.shape[0]), \
        "sut shape: {}, a.shape: {}, b.shape: {}".format(sut.shape, a.shape, b.shape)

    sut_mean_rates = mean_firing_rate(sut)
    assert tar_rate - 5e-06 < sut_mean_rates[0] < tar_rate + 5e-06, \
        "mean rate 0 should be: {}, was: {}".format(tar_rate, sut_mean_rates[0])
    assert tar_rate - 5e-06 < sut_mean_rates[1] < tar_rate + 5e-06, \
        "mean rate 0 should be: {}, was: {}".format(tar_rate, sut_mean_rates[1])

    # test neurons by columns too.
    train_10_by_100 = poisson_input(0.5 * torch.ones((10,)), t=100, N=10)
    mean_rate = mean_firing_rate(train_10_by_100)
    assert mean_rate.shape[0] == 10, "mean_rate.shape: {}".format(mean_rate.shape)
    train_20_by_40 = poisson_input(0.5 * torch.ones((20,)), t=40, N=20)
    mean_rate = mean_firing_rate(train_20_by_40)
    assert mean_rate.shape[0] == 20, "mean_rate.shape: {}".format(mean_rate.shape)


def test_sums_helper():
    s1 = poisson_input(0.5 * torch.ones((10,)), t=100, N=10)
    s2 = poisson_input(0.5 * torch.ones((10,)), t=100, N=10)

    sut = sums_helper(spikes1=s1, spikes2=s2)
    assert sut.shape[0] == 2 and sut.shape[1] == 100, "sut does not have expected shape (N=2,t=100). sut.shape: {}".format(sut.shape)


def test_van_rossum_dist():
    tar_spikes = poisson_input(1. * torch.ones((3,)), t=100, N=3)
    model_spikes = poisson_input(1. * torch.ones((3,)), t=100, N=3)
    print('num of sample model spikes: {}'.format(model_spikes.sum()))
    print('num of sample target spikes: {}'.format(tar_spikes.sum()))

    dist_poisson_spikes_tau_4 = van_rossum_dist(spikes=model_spikes, target_spikes=tar_spikes, tau=torch.tensor(4.0))
    dist_poisson_spikes_tau_20 = van_rossum_dist(spikes=model_spikes, target_spikes=tar_spikes, tau=torch.tensor(20.0))
    print('(tau=4) van rossum dist.: {}'.format(dist_poisson_spikes_tau_4))
    print('(tau=20) van rossum dist.: {}'.format(dist_poisson_spikes_tau_20))

    zeros = torch.zeros_like(tar_spikes)
    print('num of spikes in zeros: {}'.format(zeros.sum()))

    dist_zeros = van_rossum_dist(zeros.clone(), zeros, tau=torch.tensor(4.0))
    print('van rossum zero dist.: {}'.format(dist_zeros))
    assert 0 <= dist_zeros < 1e-08, "distance between silent trains should be approximately zero. was: {}".format(dist_zeros)

    distance_model_spikes_zeros_tau_vr_4 = van_rossum_dist(model_spikes, zeros, torch.tensor(4.0))
    distance_model_spikes_zeros_tau_vr_20 = van_rossum_dist(model_spikes, zeros, torch.tensor(20.0))
    print('distance_model_spikes_zeros_tau_vr_4: {}'.format(distance_model_spikes_zeros_tau_vr_4))
    print('distance_model_spikes_zeros_tau_vr_20: {}'.format(distance_model_spikes_zeros_tau_vr_20))
    assert distance_model_spikes_zeros_tau_vr_20 > distance_model_spikes_zeros_tau_vr_4, "tau 20 should result in greater loss than 4 when compared to no spikes as target"
    assert dist_poisson_spikes_tau_4 < distance_model_spikes_zeros_tau_vr_4, "some spikes should be better than none (tau 4)"
    assert dist_poisson_spikes_tau_20 < distance_model_spikes_zeros_tau_vr_20, "some spikes should be better than none (tau 20)"


def test_optimised_van_rossum():
    tau = torch.tensor(3.0)
    spikes = (torch.rand((100, 3)) > 0.85).float()

    torch_conv = torch_van_rossum_convolution(spikes, tau)

    print('no. spikes: {}, torch conv. sum: {}'.format(spikes.sum(), torch_conv.sum()))
    assert torch_conv.sum() - spikes.sum() > 0., "check torch conv. impl."


def test_different_taus_van_rossum_dist():
    t = 400; N=12
    zeros = torch.zeros((t, N))
    sample_spikes = poisson_input(0.8, t, N)
    cur_tau = torch.tensor(25.0)
    cur_dist = van_rossum_dist(sample_spikes, zeros, cur_tau)
    print('cur_tau: {:.2f}, cur_dist: {:.4f}'.format(cur_tau, cur_dist))
    for tau_i in range(24):
        prev_dist = cur_dist.clone()
        cur_tau = cur_tau - 1
        cur_dist = van_rossum_dist(sample_spikes, zeros, cur_tau)
        # print('cur_tau: {:.2f}, cur_dist: {:.4f}'.format(cur_tau, cur_dist))
        assert cur_dist < prev_dist, "decreasing tau should decrease loss when comparing poisson spikes to zero spikes"


def test_van_rossum_convolution():
    t = 400; N = 12
    tau_vr = torch.tensor(20.0)

    zeros = torch.zeros((t, N))
    sample_spikes = (poisson_input(0.8, t, N) > 0).float()
    print('no. of spikes: {}'.format(sample_spikes.sum()))

    lower_avg_rate = 0.5 * t * N
    assert sample_spikes.sum() > lower_avg_rate, "spike sum: {} should be greater than an approximate lower avg. rate: {}"\
        .format(sample_spikes.sum(), lower_avg_rate)

    conv = torch_van_rossum_convolution(sample_spikes, tau=tau_vr)
    dist = van_rossum_dist(sample_spikes, zeros, tau=tau_vr)
    dist_approx = torch.sqrt(torch.pow(conv, 2).sum() + 1e-18)
    assert dist == dist_approx, "distance: {:.4f} was not approx. dist: {:.4f}".format(dist, dist_approx)
    assert conv.sum() > sample_spikes.sum(), "conv sum should be greater than sum of spikes"

# --------------------------------------
test_mean_firing_rate()
test_sums_helper()
test_van_rossum_dist()
test_optimised_van_rossum()
test_different_taus_van_rossum_dist()
test_van_rossum_convolution()
