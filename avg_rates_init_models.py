import os

import numpy as np
import torch

import stats


def avg_rates_init_models(model_type='LIF'):
    full_folder_path = '/Users/william/repos/snn_inference/saved/plot_data/init_LIF_models/'
    if model_type == 'GLIF':
        full_folder_path = '/Users/william/repos/snn_inference/saved/plot_data/init_GLIF_models/'
    experiment_averages = {}
    # for full_folder_path in [init_LIF_models, init_GLIF_models]:
    if not full_folder_path.__contains__('.DS_Store'):
        files = os.listdir(full_folder_path)
        id = full_folder_path.split('/')[-1]
    else:
        files = []
        id = 'None'
    plot_spiketrains_files = []
    for f in files:
        if f.__contains__('plot_spiketrains_side_by_side'):
            plot_spiketrains_files.append(f)
        elif f.__contains__('plot_losses'):
            f_data = torch.load(full_folder_path + f)
            custom_title = f_data['plot_data']['custom_title']
            optimiser = custom_title.split(', ')[1].strip(' ')
            model_type = custom_title.split(',')[0].split('(')[-1]
            lr = custom_title.split(', ')[-1].strip(' =lr').strip(')')
            lfn = f_data['plot_data']['fname'].split('loss_fn_')[1].split('_tau')[0]
            # break

    print('Success! Processing exp: {}'.format(full_folder_path))
    plot_spiketrains_files.sort()  # check that alphabetically

    if not experiment_averages.__contains__(model_type):
        experiment_averages[model_type] = {
            optimiser: {lfn: {lr: {'avg_model_rate': [], 'stds_model_rates' : [],
                                   'avg_target_rate': [], 'stds_target_rates' : []}}}}
    if not experiment_averages[model_type].__contains__(optimiser):
        experiment_averages[model_type][optimiser] = {}
    if not experiment_averages[model_type][optimiser].__contains__(lfn):
        experiment_averages[model_type][optimiser][lfn] = {}
    if not experiment_averages[model_type][optimiser][lfn].__contains__(lr):
        experiment_averages[model_type][optimiser][lfn][lr] = {'avg_model_rate': [], 'stds_model_rates' : [],
                                                               'avg_target_rate': [], 'stds_target_rates' : []}

    avg_rates_model = []; stds_model_rates = []; avg_rates_target = []; stds_target_rates = []
    for exp_i in range(len(plot_spiketrains_files)):  # gen data for [0 + 11 * i]
        print('exp_i: {}'.format(exp_i))
        cur_full_path = full_folder_path + plot_spiketrains_files[exp_i]

        data = torch.load(cur_full_path)
        print('Loaded saved plot data.')

        plot_data = data['plot_data']

        cur_std_model, cur_rate_model = stats.binned_avg_firing_rate_per_neuron(plot_data['model_spikes'].detach().numpy(), bin_size=400)
        cur_std_target, cur_rate_target = stats.binned_avg_firing_rate_per_neuron(plot_data['target_spikes'].detach().numpy(), bin_size=400)

        avg_model_rate = np.mean(np.asarray(cur_rate_model), axis=0) * 1000.
        std_model_rates = np.mean(np.asarray(cur_std_model), axis=0) * 1000.
        avg_target_rate = np.mean(np.asarray(cur_rate_target), axis=0) * 1000.
        std_target_rates = np.mean(np.asarray(cur_std_target), axis=0) * 1000.

        avg_rates_model.append(avg_model_rate)
        avg_rates_target.append(avg_target_rate)
        stds_model_rates.append(std_model_rates)
        stds_target_rates.append(std_target_rates)

    # experiment_averages[model_type][optimiser][lfn][lr]['avg_model_rate'].append(np.mean(avg_rates_model))
    # experiment_averages[model_type][optimiser][lfn][lr]['stds_model_rates'].append(np.mean(stds_model_rates))
    # experiment_averages[model_type][optimiser][lfn][lr]['avg_target_rate'].append(np.mean(avg_rates_target))
    # experiment_averages[model_type][optimiser][lfn][lr]['stds_target_rates'].append(np.mean(stds_target_rates))
    return avg_rates_model, avg_rates_target, stds_model_rates, stds_target_rates


# if __name__ == "__main__":
#     main(sys.argv[1:])
