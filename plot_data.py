from data_util import load_sparse_data_matlab_format, get_spike_train_matrix
from plot import plot_spike_train


def plot_data_for_n_samples(data_exp_name, n_samples):
    node_indices, spike_times, spike_indices, qual, states = load_sparse_data_matlab_format(data_exp_name + '.mat')

    for i in range(n_samples):
        _, spike_train_108 = get_spike_train_matrix(index_last_step=int(len(spike_times)*i/n_samples),
                                                    advance_by_t_steps=4000, spike_times=spike_times,
                                                    spike_indices=spike_indices, node_numbers=node_indices)

        plot_spike_train(spike_train_108, '{} sample #{}'.format(data_exp_name, i), uuid='test_plot_data',
                         fname='{}_sample_{}'.format(data_exp_name, i))

# plot_data_for_n_samples('exp108', 15)
# plot_data_for_n_samples('exp109', 15)
# plot_data_for_n_samples('exp124', 15)
# plot_data_for_n_samples('exp126', 15)
# plot_data_for_n_samples('exp138', 15)
# plot_data_for_n_samples('exp146', 15)
plot_data_for_n_samples('exp147', 15)
