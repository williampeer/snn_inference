import os
import sys

import numpy as np
import torch

import IO
import data_util
from Models.GLIF import GLIF
from Models.Izhikevich import Izhikevich
from Models.LIF import LIF
from Models.microGIF import microGIF
from spike_train_matlab_export import simulate_and_save_model_spike_train


man_seed = 3
torch.manual_seed(man_seed)
np.random.seed(man_seed)

duration = 60 * 1000

load_fname = 'snn_model_target_GD_test'
model_class_lookup = { 'LIF': LIF, 'GLIF': GLIF, 'microGIF': microGIF, 'Izhikevich': Izhikevich }

experiments_path = '/home/william/repos/snn_inference/Test/saved/GT/'
# archive_name = 'data/'
# plot_data_path = experiments_path + 'plot_data/'
model_type_dirs = os.listdir(experiments_path)
# model_type_dirs = ['microGIF']

for model_type_str in model_type_dirs:
    if not model_type_str.__contains__("plot_data"):
        model_class = model_class_lookup[model_type_str]
        # model_class = microGIF
        exp_uids = os.listdir(experiments_path + '/' + model_type_str)
        for euid in exp_uids:
            load_data = torch.load(experiments_path + '/' + model_type_str + '/' + euid + '/' + load_fname + IO.fname_ext)
            snn = load_data['model']
            # saved_target_losses = load_data['loss']
            # num_neurons = snn.N
            cur_fname = 'nuovo_GT_spikes_mt_{}_euid_{}'.format(model_class.__name__, euid)

            if not os.path.exists(data_util.prefix + data_util.target_data_path + data_util.matlab_export + cur_fname + '.mat'):
                simulate_and_save_model_spike_train(model=snn, t=duration, exp_num=euid, model_name=model_class.__name__, fname=cur_fname)
            else:
                print('file exists. skipping..')

sys.exit()
