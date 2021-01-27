import os
import sys

import numpy as np
import torch

import plot
import stats


def main(argv):
    print('Argument List:', str(argv))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]

    load_paths = []
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_spiketrains_side_by_side01-21_21-55-15-927.pt']
    experiments_path = '/Users/william/repos/archives_snn_inference/archive 10/saved/plot_data/'
    folders = os.listdir(experiments_path)
    loss_res = {}
    for folder_path in folders:
        # print(folder_path)
        full_folder_path = experiments_path + folder_path + '/'
        if not folder_path.__contains__('.DS_Store'):
            files = os.listdir(full_folder_path)
            id = folder_path.split('-')[-1]
        else:
            files = []
            id = 'None'
        # plot_spiketrains_files = []
        # plot_losses_files = []
        for f in files:
            if f.__contains__('plot_losses'):
                f_data = torch.load(full_folder_path + f)
                custom_title = f_data['plot_data']['custom_title']
                optimiser = custom_title.split(', ')[1].strip(' ')
                model_type = custom_title.split(',')[0].split('(')[-1]
                lr = custom_title.split(', ')[-1].strip(' =lr').strip(')')
                lfn = f_data['plot_data']['fname'].split('loss_fn_')[1].split('_tau')[0]

                # plot_losses_files.append(f)
                if not loss_res.__contains__(model_type):
                    loss_res[model_type] = { optimiser: { lr: { lfn: [] } } }
                if not loss_res[model_type].__contains__(optimiser):
                    loss_res[model_type][optimiser] = {}
                if not loss_res[model_type][optimiser].__contains__(lr):
                    loss_res[model_type][optimiser][lr] = {}
                if not loss_res[model_type][optimiser][lr].__contains__(lfn):
                    loss_res[model_type][optimiser][lr][lfn] = []


                loss_res[model_type][optimiser][lr][lfn].append()
                # break

        # if len(plot_spiketrains_files) != 55:
            # print("Incomplete exp. len should be 5 exp * 11 plots. was: {}".format(len(plot_spiketrains_files)))
            # print(len(plot_spiketrains_files))
            # pass
        # else:
            print('Processing configuration: {}'.format(folder_path))
            plot_spiketrains_files.sort()  # check that alphabetically

            for train_i in [0, 4, 7, 10]:  # --> (1, 7, 13, 20)
                cur_hyperconf = '{}, {}, {}, $\\alpha={}$'.format(model_type, optimiser, lfn, lr)
                fname_prefix = model_type + '_' + optimiser + '_' + lfn
                save_fname = '{}_{}_train_iter_{}.eps'.format(fname_prefix, id, train_i)
                custom_title = 'Average firing rates, '+cur_hyperconf+', {} iteration(s)'.format(train_i)

                avg_rates_model = []; avg_rates_target = []
                for exp_i in range(5):
                    # gen data for [0 + 11 * i]
                    data = torch.load(full_folder_path + plot_spiketrains_files[train_i + 11 * exp_i])
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
