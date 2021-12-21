import os

import numpy as np
import torch

import IO
import data_util
import model_util
import plot
from Models.GLIF import GLIF
from Models.GLIF_no_cell_types import GLIF_no_cell_types
from Models.LIF import LIF
from Models.LIF_no_cell_types import LIF_no_cell_types
from Models.microGIF import microGIF
from analysis.euid_to_sleep_exp import euid_to_sleep_exp_num

man_seed = 3
torch.manual_seed(man_seed)
np.random.seed(man_seed)

model_class_lookup = { 'LIF': LIF, 'GLIF': GLIF, 'microGIF': microGIF,
                       'LIF_no_cell_types': LIF_no_cell_types, 'GLIF_no_cell_types': GLIF_no_cell_types }

experiments_path = '/media/william/p6/archive_14122021/archive/saved/sleep_data_no_types/'
experiments_path_plot_data = '/media/william/p6/archive_14122021/archive/saved/plot_data/sleep_data_no_types/'
experiments_path_sleep_data_microGIF = '/home/william/repos/snn_inference/Test/saved/sleep_data/'
experiments_path_sleep_data_microGIF_plot_data = '/home/william/repos/snn_inference/Test/saved/plot_data/sleep_data/'

# experiments_path = '/media/william/p6/archive_14122021/archive/saved/sleep_data_no_types/'
# archive_name = 'data/'
# plot_data_path = experiments_path + 'plot_data/'
# model_type_dirs = os.listdir(experiments_path)
model_type_dirs = ['LIF_no_cell_types', 'GLIF_no_cell_types', 'microGIF']
exp_folder_name = experiments_path.split('/')[-2]


def get_target_rate(exp_str):
    sleep_data_path = data_util.prefix + data_util.sleep_data_path
    sleep_data_files = ['exp108.mat', 'exp109.mat', 'exp124.mat', 'exp126.mat', 'exp138.mat', 'exp146.mat', 'exp147.mat']
    data_file = exp_str + '.mat'
    assert data_file in sleep_data_files, "exp_str: {} not found in sleep data files: {}".format(exp_str, sleep_data_files)

    node_indices, spike_times, spike_indices = data_util.load_sparse_data(sleep_data_path + data_file)
    _, target_spikes = data_util.get_spike_train_matrix(0, 12000, spike_times, spike_indices, node_indices)
    cur_mean_rate_np = get_mean_rate_for_spikes(target_spikes)
    return cur_mean_rate_np


def get_mean_rate_for_spikes(spikes):
    normalised_spike_rate = spikes.sum(dim=0) * 1000. / (spikes.shape[0])
    return np.mean(normalised_spike_rate.numpy())


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
    return np.mean(normalised_spike_rate.numpy())


def get_lfn_from_plot_data_in_folder(exp_folder):
    folder_files = os.listdir(exp_folder)
    loss_file = list(filter(lambda x: x.__contains__('plot_loss'), folder_files))[0]
    plot_data = torch.load(exp_folder + loss_file)['plot_data']
    custom_title = plot_data['custom_title']
    lfn = custom_title.split(',')[0].strip('Loss ')
    return lfn


sleep_exps = ['exp108', 'exp109', 'exp124', 'exp126', 'exp138', 'exp146', 'exp147']
result_per_exp = {}
target_rates = []
plot_exp_type = 'export_metrics'
for model_type_str in model_type_dirs:
    for exp_str in sleep_exps:
        target_rate = get_target_rate(exp_str)
        target_rates.append(target_rate)
        cur_fname = 'export_rates_{}_{}_{}.eps'.format(model_type_str, experiments_path.split('/')[-2], exp_str)

        plot_uid = model_type_str
        full_path = './figures/' + plot_exp_type + '/' + plot_uid + '/'
        # mean_rates_by_lfn = { 'frd': [], 'vrd': [], 'bernoulli_nll': [], 'poisson_nll': [] }
        if model_type_str == 'microGIF':
            result_per_exp[exp_str] = { model_type_str : { 'bernoulli_nll': [], 'poisson_nll': [] } }
        else:
            result_per_exp[exp_str] = { model_type_str : { 'frd': [], 'vrd': [] } }

    if os.path.exists(experiments_path + model_type_str):
        exp_uids = os.listdir(experiments_path + model_type_str)
        for euid in exp_uids:
            sleep_exp = euid_to_sleep_exp_num[model_type_str][euid]
            lfn = get_lfn_from_plot_data_in_folder(experiments_path_plot_data + model_type_str + '/' + euid + '/')

            load_fname = 'snn_model_target_GD_test'
            load_data = torch.load(experiments_path + '/' + model_type_str + '/' + euid + '/' + load_fname + IO.fname_ext)
            cur_model = load_data['model']
            result_per_exp[sleep_exp][model_type_str][lfn].append(get_mean_rate_for_model(cur_model))

    if os.path.exists(experiments_path_sleep_data_microGIF + model_type_str):
        exp_uids = os.listdir(experiments_path_sleep_data_microGIF + model_type_str)
        for euid in exp_uids:
            sleep_exp = euid_to_sleep_exp_num[model_type_str][euid]
            lfn = get_lfn_from_plot_data_in_folder(experiments_path_sleep_data_microGIF_plot_data + model_type_str + '/' + euid + '/')

            load_fname = 'snn_model_target_GD_test'
            load_data = torch.load(experiments_path + '/' + model_type_str + '/' + euid + '/' + load_fname + IO.fname_ext)
            cur_model = load_data['model']
            result_per_exp[sleep_exp][model_type_str][lfn].append(get_mean_rate_for_model(cur_model))

for exp_name in result_per_exp.keys():
    cur_exp_res = result_per_exp[exp_name]
    mean_rates = []; std_rates = []; xticks = []
    for model_type_name in result_per_exp[exp_name].keys():
        for lfn in cur_exp_res.keys():
            cur_mean_rate = np.mean(list(filter(lambda x: not np.isnan(x) and (x < 1.75 * target_rate and x > 0.25 * target_rate), result_per_exp[exp_name][lfn])))
            cur_std_rate = np.std(list(filter(lambda x: not np.isnan(x) and (x < 1.75 * target_rate and x > 0.25 * target_rate), result_per_exp[exp_name][lfn])))
            if np.isnan(cur_mean_rate):
                cur_mean_rate = 0.; cur_std_rate = 0.
            mean_rates.append(cur_mean_rate)
            std_rates.append(cur_std_rate)
            xticks.append('{},\n${}$'.format(model_type_name.replace('microGIF', 'miGIF').replace('mesoGIF', 'meGIF'),
                                             lfn.replace('poisson_nll', 'P_{NLL}').replace('bernoulli_nll', 'B_{NLL}')))

    model_type_fname = 'export_rates_sleep_data_{}_{}_all.eps'.format(exp_name, exp_folder_name)
    plot.bar_plot_neuron_rates(target_rates, np.asarray(mean_rates), 0., np.asarray(std_rates), plot_exp_type, 'all',
                               custom_legend=['Sleep data', 'Fitted models'],
                               fname=model_type_fname, xticks=xticks, custom_colors=['Red', 'Cyan'])


# sys.exit()
