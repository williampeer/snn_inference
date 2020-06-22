import torch
import plot
import spike_metrics


def test_plot_van_rossum_convolution():
    sample_spiketrain = 1.0 * (torch.rand((200, 3)) < 0.25)
    sample_spiketrain_2 = 1.0 * (torch.rand((200, 3)) < 0.25)
    plot.plot_spiketrain(sample_spiketrain, uuid='test_uuid', title="Sample spiketrain")
    plot.plot_spiketrains_side_by_side(sample_spiketrain, sample_spiketrain_2, 'test_uuid',
                                       title="Sample side-by-side raster plot")

    convolved = spike_metrics.torch_van_rossum_convolution(sample_spiketrain, tau=torch.tensor(5.0))
    # convolved = spike_metrics.convolve_van_rossum_using_clone(sample_spiketrain, tau=torch.tensor(5.0))
    plot.plot_neuron(convolved[:, 0], 'Spiketrain node 1 van Rossum convolved')
    plot.plot_neuron(convolved[:, 1], 'Spiketrain node 2 van Rossum convolved')
    plot.plot_neuron(convolved[:, 2], 'Spiketrain node 3 van Rossum convolved')


def test_plot_parameter_pairs_with_variance():
    sample_param_1_means = []
    sample_param_2_means = []
    sample_targets = {0: 6.5, 1: -65.0}
    # sample_param_3_means = []

    for i in range(10):
        sample_param_1_means.append(6.0 + torch.rand(1))
        sample_param_2_means.append(-66.0 + 2.0 * torch.rand(1))
        # sample_param_3_means.append(torch.tensor(9000. + i))

    fitted_param_means = {0: sample_param_1_means, 1: sample_param_2_means}

    # plot.plot_parameter_pair_with_variance(fitted_param_means[0], fitted_param_means[1], sample_targets)
    plot.plot_all_param_pairs_with_variance(param_means=fitted_param_means, target_params=sample_targets, exp_type='test',
                                            uuid='test_plots', fname='test_parameter_kdes_1', custom_title='Test plot', logger=False)

    fitted_param_means[2] = torch.rand((3, 3)).data
    # sample_targets.append(9000.)
    sample_targets[2] = torch.rand((3, 3)).data

    plot.plot_all_param_pairs_with_variance(param_means=fitted_param_means, target_params=sample_targets,
                                            exp_type='test', uuid='test_plots', fname='test_parameter_kdes_2',
                                            custom_title='Test plot', logger=False)


def test_plot_spiketrains_side_by_side():
    sample_spiketrain = 1.0 * (torch.rand((200, 3)) < 0.25)
    sample_spiketrain_2 = 1.0 * (torch.rand((200, 3)) < 0.25)

    plot.plot_spiketrains_side_by_side(sample_spiketrain, sample_spiketrain_2, 'test_uuid', title='Test plot_spiketrains_side_by_side')


test_plot_van_rossum_convolution()
test_plot_parameter_pairs_with_variance()
test_plot_spiketrains_side_by_side()
