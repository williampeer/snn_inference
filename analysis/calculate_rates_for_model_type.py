import os
import sys

import numpy as np
import torch

import IO
import data_util
import model_util
import plot
import spike_metrics
from Models.GLIF import GLIF
from Models.GLIF_no_cell_types import GLIF_no_cell_types
from Models.LIF import LIF
from Models.LIF_no_cell_types import LIF_no_cell_types
from Models.microGIF import microGIF
from spike_train_matlab_export import simulate_and_save_model_spike_train

man_seed = 3
torch.manual_seed(man_seed)
np.random.seed(man_seed)

load_fname = 'snn_model_target_GD_test'
model_class_lookup = { 'LIF': LIF, 'GLIF': GLIF, 'microGIF': microGIF,
                       'LIF_no_cell_types': LIF_no_cell_types, 'GLIF_no_cell_types': GLIF_no_cell_types }

experiments_path = '/home/william/repos/snn_inference/Test/saved/GT/'
# experiments_path = '/media/william/p6/archive_14122021/archive/saved/sleep_data_no_types/'
# archive_name = 'data/'
# plot_data_path = experiments_path + 'plot_data/'
# model_type_dirs = os.listdir(experiments_path)
# model_type_dirs = ['LIF', 'GLIF', 'microGIF']
model_type_dirs = ['LIF']


def get_target_model(model_type_str):
    GT_path = '/home/william/repos/snn_inference/Test/saved/'
    GT_model_by_type = {'LIF': '12-09_11-49-59-999',
                        'GLIF': '12-09_11-12-47-541',
                        'mesoGIF': '12-09_14-56-20-319',
                        'microGIF': '12-09_14-56-17-312'}

    GT_euid = GT_model_by_type[model_type_str]
    tar_fname = 'snn_model_target_GD_test'
    model_name = model_type_str
    if model_type_str == 'mesoGIF':
        model_name = 'microGIF'
    load_data_target = torch.load(GT_path + model_name + '/' + GT_euid + '/' + tar_fname + IO.fname_ext)
    target_model = load_data_target['model']
    return target_model


def get_mean_rate_for_model(model):
    white_noise = torch.rand((4000, model.N))
    # inputs = experiments.sine_modulated_input(white_noise)
    inputs = white_noise
    if model.__class__ is microGIF:
        _, spikes, _ = model_util.feed_inputs_sequentially_return_args(model=model, inputs=inputs.clone().detach())
    else:
        _, spikes = model_util.feed_inputs_sequentially_return_tuple(model=model, inputs=inputs.clone().detach())
    # for gen spiketrain this may be thresholded to binary values:
    spikes = torch.round(spikes).clone().detach()
    normalised_spike_rate = spikes.sum(dim=0) * 1000. / (spikes.shape[0])
    return np.mean(normalised_spike_rate.data)


for model_type_str in model_type_dirs:
    target_model = get_target_model(model_type_str)
    target_rate = get_mean_rate_for_model(target_model)
    cur_fname = 'export_rates_{}_{}_N_{}'.format(model_type_str, experiments_path.split('/')[-1], target_model.N)

    plot_exp_type = 'export_rate'
    plot_uid = model_type_str
    full_path = './figures/' + plot_exp_type + '/' + plot_uid + '/'
    if not os.path.exists(full_path + cur_fname):
        if not model_type_str.__contains__("plot_data"):
            model_class = model_class_lookup[model_type_str]
            # model_class = microGIF
            exp_uids = os.listdir(experiments_path + '/' + model_type_str)
            mean_rates = []
            for euid in exp_uids:
                load_data = torch.load(experiments_path + '/' + model_type_str + '/' + euid + '/' + load_fname + IO.fname_ext)
                cur_model = load_data['model']
                mean_rates.append(get_mean_rate_for_model(cur_model))

            plot.bar_plot_neuron_rates(mean_rates, target_rate, np.std(mean_rates), 0., plot_exp_type, plot_uid, fname=cur_fname)
    else:
        print('file exists. skipping..')


sys.exit()
