import os
import sys

import numpy as np
import torch

import IO
import data_util
import experiments
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

experiments_path = '/home/william/repos/snn_inference/Test/saved/'
experiments_path_plot_data = '/home/william/repos/snn_inference/Test/saved/plot_data/'
# experiments_path = '/media/william/p6/archive_14122021/archive/saved/sleep_data_no_types/'
# archive_name = 'data/'
# plot_data_path = experiments_path + 'plot_data/'
# model_type_dirs = os.listdir(experiments_path)
model_type_dirs = ['LIF', 'GLIF', 'microGIF']
# model_type_dirs = ['LIF']


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


def get_param_dist(model, target_model):
    model_params = model.get_parameters()
    target_params = target_model.get_parameters()
    assert len(model_params) == len(target_params), "parameter dicts should be of equal length.."
    total_mean_param_dist = 0.
    for p_v, p_k in enumerate(model_params):
        if p_k != 'w':
            p_rmse = np.sqrt(np.mean(np.power(p_v - target_params[p_k].numpy(), 2)))
            total_mean_param_dist += p_rmse

    return (total_mean_param_dist/len(model_params))


def get_param_dist_no_weights(model, target_model):
    model_params = model.get_parameters()
    target_params = target_model.get_parameters()
    assert len(model_params) == len(target_params), "parameter dicts should be of equal length.."
    total_mean_param_dist = 0.
    for p_v, p_k in enumerate(model_params):
        if p_k != 'w':
            p_rmse = torch.mean(torch.sqrt(torch.pow(p_v - target_params[p_k], 2)))
            total_mean_param_dist += p_rmse.numpy()

    return (total_mean_param_dist/len(model_params))


def get_init_param_dist_no_weights(target_model):
    model_class = target_model.__class__
    start_seed = 23
    p_dists = []
    for i in range(20):
        np.random.seed(start_seed+i)
        torch.random.manual_seed(start_seed+i)
        params_model = experiments.draw_from_uniform(model_class.parameter_init_intervals, target_model.N)
        if hasattr(target_model, 'neuron_types'):
            model = model_class(params_model, N=target_model.N, neuron_types=target_model.neuron_types)
        else:
            model = model_class(params_model, N=target_model.N)

        cur_dist = get_param_dist_no_weights(model, target_model)
        p_dists.append(cur_dist)

    return np.mean(p_dists), np.std(p_dists)


def get_lfn_from_plot_data_in_folder(exp_folder):
    folder_files = os.listdir(exp_folder)
    loss_file = list(filter(lambda x: x.__contains__('plot_loss'), folder_files))[0]
    plot_data = torch.load(exp_folder + loss_file)['plot_data']
    custom_title = plot_data['custom_title']
    lfn = custom_title.split(',')[0].strip('Loss ')
    return lfn


def get_mean_rate_for_model_helper(model):
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



plot_exp_type = 'export_p_dist'
global_fname = 'export_p_dists_{}_all.eps'.format(experiments_path.split('/')[-2])
mean_dists = []; std_dists = []; xticks = []
init_dists = []; init_dist_stds = []
for model_type_str in model_type_dirs:
    target_model = get_target_model(model_type_str)
    target_rate = get_mean_rate_for_model_helper(target_model)
    cur_fname = 'export_rates_{}_{}_N_{}.eps'.format(model_type_str, experiments_path.split('/')[-2], target_model.N)

    plot_uid = model_type_str
    full_path = './figures/' + plot_exp_type + '/' + plot_uid + '/'
    # mean_rates_by_lfn = { 'frd': [], 'vrd': [], 'bernoulli_nll': [], 'poisson_nll': [] }
    if model_type_str == 'microGIF':
        mean_dist_by_lfn = {'bernoulli_nll': [], 'poisson_nll': []}
        mean_rates_by_lfn = {'bernoulli_nll': [], 'poisson_nll': []}
    else:
        mean_dist_by_lfn = {'frd': [], 'vrd': []}
        mean_rates_by_lfn = {'frd': [], 'vrd': []}

    model_class = model_class_lookup[model_type_str]
    # model_class = microGIF
    exp_uids = os.listdir(experiments_path + '/' + model_type_str)
    for euid in exp_uids:
        lfn = get_lfn_from_plot_data_in_folder(experiments_path_plot_data + model_type_str + '/' + euid + '/')

        load_data = torch.load(experiments_path + '/' + model_type_str + '/' + euid + '/' + load_fname + IO.fname_ext)
        cur_model = load_data['model']
        mean_dist_by_lfn[lfn].append(get_param_dist(cur_model, target_model))
        mean_rates_by_lfn[lfn].append(get_mean_rate_for_model_helper(cur_model))

    for lfn in mean_dist_by_lfn.keys():
        cur_dists = []
        for r_i in range(len(mean_rates_by_lfn[lfn])):
            cur_rate = mean_rates_by_lfn[lfn][r_i]
            if not np.isnan(cur_rate) and (cur_rate < 1.75 * target_rate and cur_rate > 0.25 * target_rate):
                cur_dists.append(mean_dist_by_lfn[lfn][r_i])

        if len(cur_dists) == 0:
            cur_mean_dist = 0.; cur_std_dist = 0.
        else:
            cur_mean_dist = np.mean(cur_dists)
            cur_std_dist = np.std(cur_dists)
        mean_dists.append(cur_mean_dist)
        std_dists.append(cur_std_dist)

        init_p_dist, init_p_dist_std = get_init_param_dist_no_weights(target_model)
        init_dists.append(init_p_dist)
        init_dist_stds.append(init_p_dist_std)
        xticks.append('{},\n${}$'.format(model_type_str.replace('microGIF', 'miGIF'),
                                         lfn.replace('poisson_nll', 'P_{NLL}').replace('bernoulli_nll', 'B_{NLL}')))
# plot.bar_plot(np.asarray(mean_dists), np.asarray(std_dists), labels=xticks, exp_type=plot_exp_type, uuid='all', fname=global_fname)
# import importlib
# importlib.reload(plot)
plot.bar_plot_neuron_rates(np.asarray(mean_dists), init_dists, np.asarray(std_dists), init_dist_stds,
                           exp_type=plot_exp_type, uuid='all', fname=global_fname, xticks=xticks,
                           custom_legend=['Initial models', 'Fitted models'])

# sys.exit()
