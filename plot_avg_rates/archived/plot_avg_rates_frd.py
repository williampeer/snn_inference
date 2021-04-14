import sys

import torch
import numpy as np

import Log
import plot
import stats
from Constants import ExperimentType
from Models.LIF import LIF


def main(argv):
    print('Argument List:', str(argv))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]

    load_paths = []

    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_spiketrains_side_by_side01-20_15-07-39-872.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_spiketrains_side_by_side01-20_16-58-05-758.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_spiketrains_side_by_side01-20_18-50-09-313.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_spiketrains_side_by_side01-20_21-00-14-372.pt']

    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_spiketrains_side_by_side01-20_21-29-58-559.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_spiketrains_side_by_side01-20_23-22-00-322.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_spiketrains_side_by_side01-21_01-15-57-275.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_spiketrains_side_by_side01-21_03-24-15-559.pt']

    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_spiketrains_side_by_side01-21_03-54-53-271.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_spiketrains_side_by_side01-21_05-42-47-551.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_spiketrains_side_by_side01-21_07-34-26-653.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_spiketrains_side_by_side01-21_09-44-32-667.pt']

    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_spiketrains_side_by_side01-21_10-16-05-455.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_spiketrains_side_by_side01-21_12-04-42-764.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_spiketrains_side_by_side01-21_13-53-50-522.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_spiketrains_side_by_side01-21_16-01-58-147.pt']

    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_spiketrains_side_by_side01-21_16-33-51-540.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_spiketrains_side_by_side01-21_18-25-04-414.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_spiketrains_side_by_side01-21_20-11-17-707.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_spiketrains_side_by_side01-21_22-18-21-534.pt']

    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_spiketrains_side_by_side01-20_15-10-34-005.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_spiketrains_side_by_side01-20_16-59-20-073.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_spiketrains_side_by_side01-20_18-49-27-701.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_spiketrains_side_by_side01-20_20-58-36-479.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_spiketrains_side_by_side01-20_21-28-01-639.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_spiketrains_side_by_side01-20_23-16-53-251.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_spiketrains_side_by_side01-21_01-07-33-195.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_spiketrains_side_by_side01-21_03-17-13-691.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_spiketrains_side_by_side01-21_03-47-30-541.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_spiketrains_side_by_side01-21_05-32-58-684.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_spiketrains_side_by_side01-21_07-22-47-980.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_spiketrains_side_by_side01-21_09-31-44-074.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_spiketrains_side_by_side01-21_10-01-21-407.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_spiketrains_side_by_side01-21_11-49-36-573.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_spiketrains_side_by_side01-21_13-38-42-605.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_spiketrains_side_by_side01-21_15-45-17-838.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_spiketrains_side_by_side01-21_16-14-34-311.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_spiketrains_side_by_side01-21_18-03-38-723.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_spiketrains_side_by_side01-21_19-49-32-524.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_spiketrains_side_by_side01-21_21-55-15-927.pt']

    # load_path = '/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/'

    # fname = load_paths[0].split('/')[-2]
    # fname = fname.split('.pt')[0].replace('.', '_')
    # save_fname = 'export_{}.eps'.format(fname)
    id = '991'
    # cur_hyperconf = 'SGD, $\\alpha=0.05$'
    cur_hyperconf = 'Adam, $\\alpha=0.05$'
    fname_prefix = 'Adam_frd'
    # save_fname = '{}_{}_train_iter_1.eps'.format(fname_prefix, id)
    # save_fname = '{}_{}_train_iter_7.eps'.format(fname_prefix, id)
    # save_fname = '{}_{}_train_iter_13.eps'.format(fname_prefix, id)
    save_fname = '{}_{}_train_iter_20.eps'.format(fname_prefix, id)
    # custom_title = 'Average firing rates, '+cur_hyperconf+', 1 training iteration'
    # custom_title = 'Average firing rates, '+cur_hyperconf+', 7 training iterations'
    # custom_title = 'Average firing rates, '+cur_hyperconf+', 13 training iterations'
    custom_title = 'Average firing rates, '+cur_hyperconf+', 20 training iterations'

    avg_rates_model = []; avg_rates_target = []
    for lp in load_paths:
        data = torch.load(lp)
        print('Loaded saved plot data.')

        plot_data = data['plot_data']

        cur_std_model, cur_rate_model = stats.binned_avg_firing_rate_per_neuron(plot_data['model_spikes'].detach().numpy(), bin_size=400)
        cur_std_target, cur_rate_target = stats.binned_avg_firing_rate_per_neuron(plot_data['target_spikes'].detach().numpy(), bin_size=400)
        avg_rates_model.append(cur_rate_model.numpy())
        avg_rates_target.append(cur_rate_target.numpy())

    avg_model_rate = np.mean(np.asarray(avg_rates_model), axis=0) * 1000.
    std_model_rates = np.std(np.asarray(avg_rates_model), axis=0) * 1000.
    avg_target_rate = np.mean(np.asarray(avg_rates_target), axis=0) * 1000.
    std_target_rates = np.std(np.asarray(avg_rates_target), axis=0) * 1000.

    # plot.plot_avg_losses(avg_train_loss, std_train_loss, avg_test_loss, std_test_loss, uuid='export',
    #                      custom_title=custom_title, fname=save_fname)
    plot.bar_plot_neuron_rates(avg_model_rate, avg_target_rate, std_model_rates, std_target_rates, bin_size=400,
                               exp_type=plot_data['exp_type'], uuid='export',
                               fname='export_rate_plot_{}'.format(save_fname), custom_title=custom_title)


if __name__ == "__main__":
    main(sys.argv[1:])
