import sys

import torch

import Log
import plot
import stats


def main(argv):
    print('Argument List:', str(argv))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]

    load_path = None
    # load_path = './saved/plot_data/06-24_10-19-53-648/plot_spiketrains_side_by_side06-24_10-20-26-872.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive inf 3006-1747/saved/plot_data/06-30_15-27-30-586/plot_spiketrains_side_by_side06-30_15-42-07-842.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive m3 3006-1854/saved/plot_data/06-30_15-37-15-989/plot_all_param_pairs_with_variance06-30_15-50-51-290.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive inf 3006-1747/saved/plot_data/06-30_14-58-39-186/plot_all_param_pairs_with_variance06-30_15-09-44-546.pt'

    # load_path = '/Users/william/repos/archives_snn_inference/archive inf 3006-1747/saved/plot_data/06-30_14-58-39-186/plot_all_param_pairs_with_variance06-30_15-09-44-546.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive inf 3006-1747/saved/plot_data/06-30_14-58-39-186/plot_spiketrains_side_by_side06-30_15-08-35-978.pt'

    # load_path = '/Users/william/repos/archives_snn_inference/archive inf 3006-1747/saved/plot_data/06-30_14-59-53-206/plot_all_param_pairs_with_variance06-30_15-26-30-343.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive inf 3006-1747/saved/plot_data/06-30_14-59-53-206/plot_spiketrains_side_by_side06-30_15-25-20-409.pt'

    # load_path = '/Users/william/repos/archives_snn_inference/archive inf 0107-1313/saved/plot_data/07-01_10-27-06-734/plot_all_param_pairs_with_variance07-01_11-03-36-784.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive inf 0107-1313/saved/plot_data/07-01_10-27-06-734/plot_spiketrains_side_by_side07-01_11-03-35-931.pt'

    # load_path = '/Users/william/repos/archives_snn_inference/archive inf 0107-1619/saved/plot_data/07-01_11-20-27-519/plot_all_param_pairs_with_variance07-01_11-37-45-060.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive inf 0107-1619/saved/plot_data/07-01_11-20-27-519/plot_spiketrains_side_by_side07-01_11-37-43-267.pt'

    # load_path = '/Users/william/repos/archives_snn_inference/archive inf 0107-1619/saved/plot_data/07-01_11-29-17-136/plot_all_param_pairs_with_variance07-01_11-50-23-494.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive inf 0107-1619/saved/plot_data/07-01_11-29-17-136/plot_spiketrains_side_by_side07-01_11-50-21-891.pt'

    # load_path = '/Users/william/repos/archives_snn_inference/archive inf 0107-1937/saved/plot_data/07-01_15-48-18-521/plot_spiketrains_side_by_side07-01_15-55-15-508.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive inf 0107-1937/saved/plot_data/07-01_15-48-18-521/plot_all_param_pairs_with_variance07-01_15-55-16-869.pt'

    # load_path = '/Users/william/repos/archives_snn_inference/archive inf 0107-morning/saved/plot_data/06-30_19-07-48-687/plot_all_param_pairs_with_variance06-30_22-33-56-245.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive inf 0107-morning/saved/plot_data/06-30_19-07-48-687/plot_spiketrains_side_by_side06-30_22-33-55-374.pt'

    # load_path = '/Users/william/repos/archives_snn_inference/archive inf 0207-0750/saved/plot_data/07-01_15-49-26-138/plot_all_param_pairs_with_variance07-01_16-15-36-924.pt'
    load_path = '/Users/william/repos/archives_snn_inference/archive inf 0207-0750/saved/plot_data/07-01_15-49-26-138/plot_spiketrains_side_by_side07-01_16-15-35-707.pt'

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
    fname = fname.split('.pt')[0].replace('.', '_')
    save_fname = 'export_{}.eps'.format(fname)
    print('Saving to fname: {}'.format(save_fname))
    if plot_fn == 'plot_spiketrains_side_by_side':
        plot.plot_spiketrains_side_by_side(plot_data['model_spikes'], plot_data['target_spikes'], 'export',
                                           plot_data['exp_type'], 'Spike trains (Poisson input)', save_fname, export=True)

        # dev
        s1 = plot_data['model_spikes'].detach().numpy()
        s2 = plot_data['target_spikes'].detach().numpy()
        bin_size = 400
        corrs = stats.spike_train_corr_new(s1=s1, s2=s2, bin_size=bin_size)
        plot.heatmap_spike_train_correlations(corrs[12:, :12], axes=['Fitted model', 'Sleep model'], exp_type=plot_data['exp_type'], uuid='export',
                                              fname='heatmap_bin_{}_{}'.format(bin_size, save_fname), bin_size=bin_size)
        std1, r1 = stats.binned_avg_firing_rate_per_neuron(s1, bin_size=bin_size)
        std2, r2 = stats.binned_avg_firing_rate_per_neuron(s2, bin_size=bin_size)
        plot.bar_plot_neuron_rates(r1, r2, std1, std2, bin_size=bin_size, exp_type=plot_data['exp_type'], uuid='export',
                                   fname='rate_plot_bin_{}_{}'.format(bin_size, save_fname))

    elif plot_fn == 'plot_all_param_pairs_with_variance':

        plot.plot_all_param_pairs_with_variance(param_means=plot_data['param_means'], target_params=plot_data['target_params'],
                                                exp_type=plot_data['exp_type'], uuid='export',
                                                fname='export_{}'.format(save_fname),
                                                custom_title='',
                                                logger=Log.Logger('test'))



if __name__ == "__main__":
    main(sys.argv[1:])
