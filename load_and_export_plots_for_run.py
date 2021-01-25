import sys

import torch
import numpy as np

import Log
import plot
import stats
from Constants import ExperimentType
from Models.GLIF import GLIF
from Models.LIF import LIF


def main(argv):
    print('Argument List:', str(argv))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]

    # load_path = None
    # load_path = '/Users/william/repos/archives_snn_inference/archive 6/saved/plot_data/' # ++
    # load_path = '/Users/william/repos/archives_snn_inference/archive 7/saved/plot_data/01-20_15-05-33-991/plot_all_param_pairs_with_variance01-21_22-08-07-671.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive 7/saved/plot_data/01-21_22-08-07-832/plot_all_param_pairs_with_variance01-23_05-11-38-927.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive 7/saved/plot_data/01-23_05-11-39-091/plot_all_param_pairs_with_variance01-24_13-44-20-205.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive 7/saved/plot_data/01-20_15-07-44-795/plot_all_param_pairs_with_variance01-21_18-12-56-942.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive 7/saved/plot_data/01-21_18-12-56-967/plot_all_param_pairs_with_variance01-22_21-37-40-416.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive 7/saved/plot_data/01-22_21-37-40-455/plot_all_param_pairs_with_variance01-24_02-32-42-934.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive 7/saved/plot_data/01-21_14-52-28-339/plot_all_param_pairs_with_variance01-22_11-04-09-162.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive 7/saved/plot_data/01-22_11-04-09-239/plot_all_param_pairs_with_variance01-23_07-35-14-081.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive 7/saved/plot_data/01-23_07-35-14-176/plot_all_param_pairs_with_variance01-24_05-06-49-285.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive 7/saved/plot_data/01-24_05-06-49-443/plot_all_param_pairs_with_variance01-25_02-34-51-034.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive 7/saved/plot_data/01-20_15-04-49-336/plot_all_param_pairs_with_variance01-21_22-30-30-276.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive 7/saved/plot_data/01-22_21-37-40-455/plot_all_param_pairs_with_variance01-24_02-32-42-934.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive 7/saved/plot_data/01-23_06-07-29-245/plot_all_param_pairs_with_variance01-24_15-14-04-262.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive 7/saved/plot_data/01-20_15-06-33-420/plot_all_param_pairs_with_variance01-21_16-48-38-652.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive 7/saved/plot_data/01-21_16-48-38-729/plot_all_param_pairs_with_variance01-22_18-58-47-432.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive 7/saved/plot_data/01-22_18-58-47-549/plot_all_param_pairs_with_variance01-23_22-37-29-541.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive 7/saved/plot_data/01-23_22-37-29-636/plot_all_param_pairs_with_variance01-25_03-12-05-008.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive 7/saved/plot_data/01-20_15-05-16-907/plot_all_param_pairs_with_variance01-21_15-38-22-783.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive 7/saved/plot_data/01-21_15-38-22-857/plot_all_param_pairs_with_variance01-22_16-37-22-985.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive 7/saved/plot_data/01-22_16-37-23-061/plot_all_param_pairs_with_variance01-23_18-54-27-058.pt'
    load_path = '/Users/william/repos/archives_snn_inference/archive 7/saved/plot_data/01-23_18-54-27-129/plot_all_param_pairs_with_variance01-24_22-04-32-965.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive 7/saved/plot_data/'

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
        plot.heatmap_spike_train_correlations(corrs[12:, :12], axes=['Fitted model', 'Target model'], exp_type=plot_data['exp_type'], uuid='export',
                                              fname='heatmap_bin_{}_{}'.format(bin_size, save_fname), bin_size=bin_size)
        std1, r1 = stats.binned_avg_firing_rate_per_neuron(s1, bin_size=bin_size)
        std2, r2 = stats.binned_avg_firing_rate_per_neuron(s2, bin_size=bin_size)
        plot.bar_plot_neuron_rates(r1, r2, std1, std2, bin_size=bin_size, exp_type=plot_data['exp_type'], uuid='export',
                                   fname='rate_plot_bin_{}_{}'.format(bin_size, save_fname))

    elif plot_fn == 'plot_all_param_pairs_with_variance':
        print('target params', plot_data['target_params'])
        fixed_exp_params = {}
        for i in range(1,len(plot_data['param_means'])):
            cur_p = np.array(plot_data['param_means'][i])
            s = cur_p.shape
            assert len(s) == 3, "for reshaping length should be 3"
            fixed_exp_params[i-1] = np.reshape(np.array(cur_p), (s[0], s[2]))

        for key in plot_data['target_params']:
            plot_data['target_params'][key] = [plot_data['target_params'][key]]

        plot.plot_all_param_pairs_with_variance(param_means=fixed_exp_params, target_params=plot_data['target_params'],
                                                param_names=LIF.parameter_names[1:],
                                                # exp_type=plot_data['exp_type'],
                                                exp_type=ExperimentType.DataDriven.name,
                                                uuid=plot_data['uuid'],
                                                fname='export_{}'.format(save_fname),
                                                custom_title='',
                                                logger=Log.Logger('test'),
                                                export_flag=True)


if __name__ == "__main__":
    main(sys.argv[1:])
