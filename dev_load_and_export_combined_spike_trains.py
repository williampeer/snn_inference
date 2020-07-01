import sys

import torch

import plot
import stats
from Test.TestLog import TestLogger


def main(argv):
    print('Argument List:', str(argv))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]

    # sm_path = '/Users/william/data/sleep_data/LIF_sleep_model/LIF_sleep_model.pt'
    m1_path = '/Users/william/repos/archives_snn_inference/archive inf 0107-1313/saved/plot_data/07-01_10-27-06-734/plot_spiketrains_side_by_side07-01_11-03-35-931.pt'
    # m2_path = '/Users/william/repos/archives_snn_inference/archive inf 0107-1619/saved/plot_data/07-01_11-29-17-136/plot_spiketrains_side_by_side07-01_11-50-21-891.pt'
    m2_path = '/Users/william/repos/archives_snn_inference/archive inf 0107-1937/saved/plot_data/07-01_15-48-18-521/plot_spiketrains_side_by_side07-01_15-55-15-508.pt'

    data1 = torch.load(m1_path)
    data2 = torch.load(m2_path)
    print('Loaded saved plot data.')

    pd1 = data1['plot_data']
    pd2 = data2['plot_data']

    save_fname = 'export_combined_fitted_and_retrieved_LIF_complex_to_LIF_sleep_model.eps'
    print('Saving to fname: {}'.format(save_fname))

    # plot.plot_spiketrains_side_by_side(plot_data['model_spikes'], plot_data['target_spikes'], 'export',
    #                                    plot_data['exp_type'], 'Spike trains (Poisson input)', save_fname, export=True)

    # dev
    t = pd1['target_spikes']
    s1 = pd1['model_spikes']
    s2 = pd2['model_spikes']
    bin_size = 400
    plot.plot_all_spiketrains([t, s1, s2], 'export', title='Spike trains (Poisson input)', fname=save_fname,
                              legend=['Sleep model', 'Fitted model', 'Retrieved model\n(fitted to fitted)'])

    # corrs = stats.spike_train_corr_new(s1=s1, s2=s2, bin_size=bin_size)
    # plot.heatmap_spike_train_correlations(corrs[12:, :12], axes=['Fitted model', 'Sleep model'], exp_type='default', uuid='export',
    #                                       fname='heatmap_bin_{}_{}'.format(bin_size, save_fname), bin_size=bin_size)
    # std1, r1 = stats.binned_avg_firing_rate_per_neuron(s1, bin_size=bin_size)
    # std2, r2 = stats.binned_avg_firing_rate_per_neuron(s2, bin_size=bin_size)
    # plot.bar_plot_neuron_rates(r1, r2, std1, std2, bin_size=bin_size, exp_type='default', uuid='export',
    #                            fname='rate_plot_bin_{}_{}'.format(bin_size, save_fname))


if __name__ == "__main__":
    main(sys.argv[1:])
