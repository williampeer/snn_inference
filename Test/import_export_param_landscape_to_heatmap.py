import sys

import numpy as np
import torch

import IO
import plot
from Models.microGIF import microGIF

A_coeffs = [torch.randn((4,))]
phase_shifts = [torch.rand((4,))]
input_types = [1, 1, 1, 1]
t = 1200
num_steps = 100

archive_path = '/home/william/repos/archives_snn_inference/archive_1911/archive/saved/plot_data/test/'
# specific_plot_file = 'plot_parameter_landscape11-18_15-50-34-535.pt'
# specific_plot_file = 'plot_parameter_landscape11-18_16-59-03-748.pt'
specific_plot_file = 'plot_parameter_landscape11-18_18-08-50-429.pt'
# specific_plot_file = 'plot_parameter_landscape11-18_19-18-25-950.pt'

load_data = torch.load(archive_path + specific_plot_file)
save_data = load_data['plot_data']
# data = {'p1s': p1s, 'p2s': p2s, 'summary_statistic': summary_statistic,
#             'p1_name': p1_name, 'p2_name': p2_name, 'statistic_name': statistic_name,
#             'exp_type': exp_type, 'uuid': uuid, 'fname': fname}

N_dim = int(np.sqrt(len(save_data['p1s'])))  # assuming equal length of p1s and p2s
heat_mat = np.zeros((N_dim, N_dim))
summary_norm_const = np.max(save_data['summary_statistic'])
p1_last = save_data['p1s'][-1]
p2_last = save_data['p2s'][-1]
for i in range(len(save_data['p1s'])):
    # x_ind = int(save_data['p1s'][i] / p1_last)
    # y_ind = int(save_data['p2s'][i] / p2_last)
    x_ind = i % N_dim
    y_ind = int(i/N_dim)
    heat_mat[x_ind, y_ind] = save_data['summary_statistic'][i] / summary_norm_const

# ---------------- target data feature request from Arno ------------------
prev_timestamp = '11-16_11-21-13-903'
fname = 'snn_model_target_GD_test'
load_data = torch.load(IO.PATH + microGIF.__name__ + '/' + prev_timestamp + '/' + fname + IO.fname_ext)
snn_target = load_data['model']
target_params = snn_target.get_parameters()
tar_p1 = target_params[save_data['p1_name']].numpy()
tar_p2 = target_params[save_data['p2_name']].numpy()
t_p1_index = int(N_dim * (np.mean(tar_p1) / p1_last))
t_p2_index = int(N_dim * (np.mean(tar_p2) / p2_last))
target_coords = [t_p1_index, t_p2_index]
# ---------------- target data feature request from Arno ------------------

axes = ['${}$'.format(save_data['p1_name']), '${}$'.format(save_data['p2_name'])]
exp_type = 'test'; uuid = 'export_p_landscape_2d'
plot.plot_heatmap(heat_mat, axes, exp_type, uuid, fname='test_export_2d_heatmap_{}_{}.png'.format(save_data['p1_name'], save_data['p2_name']),
                  target_coords=target_coords)

# sys.exit(0)
