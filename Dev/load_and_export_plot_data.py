import sys

import torch
import numpy as np

import Log
import plot
import stats
from Constants import ExperimentType
from Models.GLIF import GLIF


def main(argv):
    print('Argument List:', str(argv))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]

    load_path = None
    # load_path = '/home/william/repos/archives_snn_inference/archive (4)/saved/plot_data/09-28_09-09-58-741/plot_all_param_pairs_with_variance09-28_15-31-47-588.pt'
    load_path = '/home/william/repos/archives_snn_inference/archive (4)/saved/plot_data/09-28_14-13-16-244/plot_all_param_pairs_with_variance09-28_19-20-27-231.pt'

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
                                                param_names=GLIF.parameter_names[1:],
                                                # exp_type=plot_data['exp_type'],
                                                exp_type=ExperimentType.DataDriven.name,
                                                uuid=plot_data['uuid'],
                                                fname='export_{}'.format(save_fname),
                                                custom_title='',
                                                logger=Log.Logger('test'),
                                                export_flag=True)



if __name__ == "__main__":
    main(sys.argv[1:])
