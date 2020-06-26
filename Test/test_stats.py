import stats
from experiments import poisson_input


def test_spike_train_correlation():
    t = 12000; N=12; bin_size = 400
    s1 = (poisson_input(0.6, t=t, N=N) > 0).float()
    s2 = (poisson_input(0.6, t=t, N=N) > 0).float()

    corrs_vars = stats.spike_train_correlation(s1, s2, bin_size=bin_size)
    assert corrs_vars.shape[0] == N and corrs_vars.shape[1] == N, \
        "spiketrain correlations should be NxN. correlations shape: {}, N: {}".format(corrs_vars.shape, N)

    for corr_i in range(corrs_vars.shape[0]):
        for corr_j in range(corrs_vars.shape[1]):
            assert corrs_vars[corr_i][corr_j] < 0.5, \
                "should have no strongly correlated bins for poisson input. i: {}, j: {}, corr: {}"\
                    .format(corr_i, corr_j, corrs_vars[corr_i][corr_j])


def test_plot_spike_train_correlations():
    pass

test_spike_train_correlation()
test_plot_spike_train_correlations()
