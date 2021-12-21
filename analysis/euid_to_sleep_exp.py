import os
import sys

import numpy as np
import torch

import IO
import data_util
import model_util
from Models.microGIF import microGIF


def get_lfn_from_plot_data_in_folder(exp_folder):
    folder_files = os.listdir(exp_folder)
    loss_file = list(filter(lambda x: x.__contains__('plot_loss'), folder_files))[0]
    plot_data = torch.load(exp_folder + loss_file)['plot_data']
    custom_title = plot_data['custom_title']
    lfn = custom_title.split(',')[0].strip('Loss ')
    return lfn


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


experiments_path = '/media/william/p6/archive_14122021/archive/saved/sleep_data_no_types/'
# experiments_path_plot_data = '/media/william/p6/archive_14122021/archive/saved/plot_data/sleep_data_no_types/'

experiments_path_sleep_data_microGIF = '/home/william/repos/snn_inference/Test/saved/sleep_data/'
# experiments_path_plot_sleep_data_microGIF = '/home/william/repos/snn_inference/Test/saved/plot_data/sleep_data/'

sleep_exps = ['exp108', 'exp109', 'exp124', 'exp126', 'exp138', 'exp146', 'exp147']
sleep_data_approx_rates = []
for sleep_exp in sleep_exps:
    sleep_data_rate = get_target_rate(sleep_exp)
    sleep_data_approx_rates.append(sleep_data_rate)

euid_to_sleep_exp_num = {}
model_type_dirs = ['LIF_no_cell_types', 'GLIF_no_cell_types']
for model_type_str in model_type_dirs:
    euid_to_sleep_exp_num[model_type_str] = {}
    exp_uids = os.listdir(experiments_path + model_type_str)
    euid_num = 0
    assert len(exp_uids) == 2*7*20, "more than 2*7*20 exps. please add logic to account for exps."
    for euid in exp_uids:
        # lfn = get_lfn_from_plot_data_in_folder(experiments_path_plot_data + model_type_str + '/' + euid + '/')
        # load_fname = 'snn_model_target_GD_test'
        # load_data = torch.load(experiments_path + '/' + model_type_str + '/' + euid + '/' + load_fname + IO.fname_ext)
        # cur_model = load_data['model']
        # mean_model_rate = get_mean_rate_for_model(cur_model)
        # dist_to_extimated_exp_rate = np.sqrt(np.power(mean_model_rate-sleep_data_approx_rates[estimated_exp_num], 2))
        # for exp_i_other in range(len(sleep_data_approx_rates)):
        #     if exp_i_other != estimated_exp_num:
        #         dist_to_other = np.sqrt(np.power(mean_model_rate - sleep_data_approx_rates[exp_i_other], 2))
        #         assert dist_to_extimated_exp_rate < (dist_to_other), "dist to exp rate should be lower than to other exps."
        estimated_exp_num = int(np.floor(euid_num / 20)) % 7
        euid_to_sleep_exp_num[model_type_str][euid] = sleep_exps[estimated_exp_num]
        euid_num += 1

euid_to_sleep_exp_num['microGIF'] = {}
exp_uids = os.listdir(experiments_path_sleep_data_microGIF + 'microGIF')
euid_num = 0
assert len(exp_uids) == 2*7*20, "more than 2*7*20 exps. please add logic to account for exps."
for euid in exp_uids:
    estimated_exp_num = int(np.floor(euid_num / 20)) % 7
    euid_to_sleep_exp_num['microGIF'][euid] = sleep_exps[estimated_exp_num]
    euid_num += 1
