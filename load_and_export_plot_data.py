import sys

import torch

import plot
import stats


def main(argv):
    print('Argument List:', str(argv))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]

    load_path = None
    # load_path = './saved/plot_data/06-24_10-19-53-648/plot_spiketrains_side_by_side06-24_10-20-26-872.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive inf 3006-1747/saved/plot_data/06-30_15-27-30-586/plot_spiketrains_side_by_side06-30_15-42-07-842.pt'
    load_path = '/Users/william/repos/archives_snn_inference/archive m3 3006-1854/saved/plot_data/06-30_15-37-15-989/plot_spiketrains_side_by_side06-30_15-50-50-112.pt'

    for i, opt in enumerate(opts):
        if opt == '-h':
            print('load_and_export_plot_data.py -p <path>')
            sys.exit()
        elif opt in ("-p", "--path"):
            load_path = args[i]

    if load_path is None:
        print('No path to load model from specified.')
        sys.exit(1)

    data = torch.load(load_path)
    print('Loaded saved plot data.')

    plot_data = data['plot_data']
    plot_fn = data['plot_fn']

    fname = load_path.split('/')[-1]
    fname = fname.split('.pt')[0]
    save_fname = 'export_{}.eps'.format(fname)
    print('Saving to fname: {}'.format(save_fname))
    if plot_fn == 'plot_spiketrains_side_by_side':
        plot.plot_spiketrains_side_by_side(plot_data['model_spikes'], plot_data['target_spikes'], 'export',
                                           plot_data['exp_type'], plot_data['title'], save_fname, export=True)

        # dev
        s1 = plot_data['model_spikes'].detach().numpy()
        s2 = plot_data['target_spikes'].detach().numpy()
        corrs_400 = stats.spike_train_correlation(s1=s1, s2=s2, bin_size=400)
        plot.heatmap_spike_train_correlations(corrs_400, axes=['Model neurons', 'Target neurons'], exp_type=plot_data['exp_type'], uuid='export',
                                              fname='heatmap_bin_400_{}'.format(save_fname))
        std1, r1 = stats.binned_avg_firing_rate_per_neuron(s1, bin_size=400)
        std2, r2 = stats.binned_avg_firing_rate_per_neuron(s2, bin_size=400)
        plot.bar_plot_neuron_rates(r1, r2, std1, std2, bin_size=400, exp_type=plot_data['exp_type'], uuid='export',
                                   fname='rate_plot_bin_400_{}'.format(save_fname))


if __name__ == "__main__":
    main(sys.argv[1:])
