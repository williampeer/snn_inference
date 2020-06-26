from experiments import poisson_input
from plot import bar_plot_neuron_rates
from stats import binned_avg_firing_rate_per_neuron


def test_bar_plot_neuron_rates():
    t = 12000; N = 12; bin_size=400
    s1 = (poisson_input(0.5, t=t, N=N) > 0).float()
    s2 = (poisson_input(0.5, t=t, N=N) > 0).float()

    std1, r1 = binned_avg_firing_rate_per_neuron(s1, bin_size)
    std2, r2 = binned_avg_firing_rate_per_neuron(s2, bin_size)
    assert r1.shape[0] == N and r2.shape[0] == N, "should be rate per neuron. r1.shape: {}, N: {}".format(r1.shape, N)

    bar_plot_neuron_rates(r1, r2, std1, std2, bin_size=bin_size,
                          exp_type='default', uuid='test_bar_plot_rates', fname='test_bar_plot_neuron_rates')


test_bar_plot_neuron_rates()
