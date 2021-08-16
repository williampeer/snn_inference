import os

import numpy as np
import torch
from torch import FloatTensor as FT
from torch import tensor as T

from Models.GLIF import GLIF
from Models.LIF import LIF
from Models.LIF_ASC import LIF_ASC
from Models.LIF_R import LIF_R
from Models.LIF_R_ASC import LIF_R_ASC
from experiments import draw_from_uniform, zip_dicts
from plot import bar_plot_pair_custom_labels, bar_plot_two_grps
from TargetModels.TargetModels import *

class_lookup = {'LIF': LIF, 'LIF_R': LIF_R, 'LIF_ASC': LIF_ASC, 'LIF_R_ASC': LIF_R_ASC, 'GLIF': GLIF}
target_fn_lookup = {'LIF': lif_continuous_ensembles_model_dales_compliant,
                    'LIF_R': lif_r_continuous_ensembles_model_dales_compliant,
                    'LIF_ASC': lif_asc_continuous_ensembles_model_dales_compliant,
                    'LIF_R_ASC': lif_r_asc_continuous_ensembles_model_dales_compliant,
                    'GLIF': glif_continuous_ensembles_model_dales_compliant}


def get_init_params(model_class, exp_num, N=12):
    torch.manual_seed(exp_num)
    np.random.seed(exp_num)

    w_mean = 0.3;
    w_var = 0.2;
    programmatic_neuron_types = torch.ones((N,))
    for n_i in range(int(2 * N / 3), N):
        programmatic_neuron_types[n_i] = -1
    neuron_types = programmatic_neuron_types
    rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((N, N))
    for i in range(len(neuron_types)):
        if neuron_types[i] == -1:
            rand_ws[i, :] = -torch.abs(FT(rand_ws[i, :]))
        elif neuron_types[i] == 1:
            rand_ws[i, :] = torch.abs(FT(rand_ws[i, :]))
        else:
            raise NotImplementedError()

    return zip_dicts({'w': rand_ws}, draw_from_uniform(model_class.parameter_init_intervals, N))


def euclid_dist(p1, p2):
    # print('p1:', p1)
    # print('p1.shape:', p1.shape)
    # print('p2:', p2)
    # sqrt((s1 - s2) ** 2)
    return np.sqrt(np.power((p1 - p2), 2).sum()) / len(p1)

# all_exps_path = '/Users/william/repos/archives_snn_inference/archive 13/saved/plot_data/'
# all_exps_path = '/home/william/repos/archives_snn_inference/archive (5)/saved/plot_data/'
# all_exps_path = '/home/william/repos/archives_snn_inference/archive/saved/plot_data/'
all_exps_path = '/home/william/repos/archives_snn_inference/archive_0908/archive/saved/plot_data/'
folders = os.listdir(all_exps_path)
experiment_averages = {}
optim_to_include = 'RMSprop'
# res_per_exp = {}
for exp_folder in folders:
    full_folder_path = all_exps_path + exp_folder + '/'

    if not exp_folder.__contains__('.DS_Store'):
        files = os.listdir(full_folder_path)
        id = exp_folder.split('-')[-1]
    else:
        files = []
        id = 'None'

    param_files = []; optimiser = None; model_type = ''
    for f in files:
        if f.__contains__('plot_all_param_pairs_with_variance'):
            param_files.append(f)
        elif optimiser is None and f.__contains__('plot_losses'):
            f_data = torch.load(full_folder_path + f)
            custom_title = f_data['plot_data']['custom_title']
            spf = custom_title.split('spf=')[-1].split(')')[0]
            optimiser = custom_title.split(', ')[1].strip(' ')
            model_type = custom_title.split(',')[0].split('(')[-1]
            lr = custom_title.split(', ')[-1].strip(' =lr').strip(')')
            lfn = f_data['plot_data']['fname'].split('loss_fn_')[1].split('_tau')[0]

    # assert len(param_files) == 1, "should only be one plot_all_param_pairs_with_variance-file per folder. len: {}".format(len(param_files))
    # if model_type == 'LIF' and len(param_files) == 1:
    # if model_type == 'GLIF' and optimiser == 'SGD' and len(param_files) == 1:
    if optimiser == optim_to_include and model_type in ['LIF_R', 'LIF_R_ASC', 'GLIF'] and spf == 'None' and len(param_files) == 1:
        print('Succes! Processing exp: {}'.format(exp_folder + '/' + param_files[0]))
        exp_data = torch.load(full_folder_path + param_files[0])
        # param_names = exp_data['plot_data']['param_names']
        param_names = class_lookup[model_type].parameter_names
        m_p_by_exp = exp_data['plot_data']['param_means']
        if(len(m_p_by_exp)>0):
            model_N = m_p_by_exp[1][0][0].shape[0]
            # t_p_by_exp = list(exp_data['plot_data']['target_params'].values())

            # config = '{}_{}_{}_{}'.format(model_type, optimiser, lfn, lr.replace('.', '_'))
            config = '{}_{}_{}'.format(model_type, optimiser, lfn)
            # distances = {}
            # stds = {}
            # distances_init = {}
            # stds_init = {}
            if not experiment_averages.__contains__(config):
                experiment_averages[config] = { 'dist' : {}, 'std': {}, 'init_dist': {}, 'init_std': {}}
                for k in range(len(m_p_by_exp)):
                    experiment_averages[config]['dist'][param_names[k]] = []
                    experiment_averages[config]['std'][param_names[k]] = []
                    experiment_averages[config]['init_dist'][param_names[k]] = []
                    experiment_averages[config]['init_std'][param_names[k]] = []

            for p_i in range(len(m_p_by_exp)):
                per_exp = []
                for e_i in range(len(m_p_by_exp[p_i])):
                    init_model_params = get_init_params(class_lookup[model_type], e_i, N=model_N)
                    # if(param_names[p_i] in init_model_params.keys()):
                    target_model = target_fn_lookup[model_type](random_seed=3 + e_i, N=model_N)
                    t_p_by_exp = target_model.params_wrapper()
                    c_d = euclid_dist(init_model_params[param_names[p_i]].numpy(), t_p_by_exp[p_i])
                    per_exp.append(c_d)
                experiment_averages[config]['init_dist'][param_names[p_i]].append(np.mean(per_exp))
                experiment_averages[config]['init_std'][param_names[p_i]].append(np.std(per_exp))

            for p_i in range(len(m_p_by_exp)):
                per_exp = []
                for e_i in range(len(m_p_by_exp[p_i])):
                    target_model = target_fn_lookup[model_type](random_seed=3 + e_i, N=model_N)
                    t_p_by_exp = target_model.params_wrapper()
                    c_d = euclid_dist(m_p_by_exp[p_i][e_i][0], t_p_by_exp[p_i])
                    per_exp.append(c_d)
                experiment_averages[config]['dist'][param_names[p_i]].append(np.mean(per_exp))
                experiment_averages[config]['std'][param_names[p_i]].append(np.std(per_exp))


# unpack
exp_avg_ds = []; exp_avg_stds = []; exp_avg_init_ds = []; exp_avg_init_stds = []
keys_list = list(experiment_averages.keys())
keys_list.sort()
labels = []
for k_i, k_v in enumerate(keys_list):
    model_type = k_v.split('_{}'.format(optim_to_include))[0]
    param_names = class_lookup[model_type].parameter_names
    label_param_names = map(lambda x: '${}$'.format(x.replace('delta_theta_', '\delta\\theta_').replace('delta_V', '\delta_V').replace('tau', '\\tau')), param_names)
    # if not (k_v.__contains__('vrdfrda') or k_v.__contains__('pnll')):
    # if not (k_v.__contains__('vrdfrda') or k_v.__contains__('pnll')):
    if True:
    # if k_v not in ['LIF_Adam_vrdfrda_0_05', 'LIF_Adam_pnll_0_05']:
        labels.append(k_v
                      .replace('LIF_SGD_', 'LIF\n')
                      .replace('LIF_R_SGD_', 'R\n')
                      .replace('LIF_ASC_SGD_', 'A\n')
                      .replace('LIF_R_ASC_SGD_', 'R_A\n')
                      .replace('GLIF_SGD_', 'GLIF\n')
                      .replace('LIF_RMSprop_', 'LIF\n')
                      .replace('LIF_R_RMSprop_', 'R\n')
                      .replace('LIF_ASC_RMSprop_', 'A\n')
                      .replace('LIF_R_ASC_RMSprop_', 'R_A\n')
                      .replace('GLIF_RMSprop_', 'GLIF\n')
                      # .replace('_', '\n')
                      # .replace('LIF_Adam_', '').replace('LIF_SGD_', '').replace('_', '\n')
                      # .replace('frdvrda', '$d_A$').replace('frdvrd', '$d_C$')
                      .replace('FIRING_RATE_DIST', '$d_F$')
                      .replace('RATE_PCC_HYBRID', '$d_P$')
                      .replace('VAN_ROSSUM_DIST', '$d_V$')
                      .replace('MSE', '$mse$'))
        print('processing exp results for config: {}'.format(k_v))
        flat_ds = []; flat_stds = []
        for d_i, d in enumerate(experiment_averages[k_v]['dist'].values()):
            flat_ds.append(d[0])
        for s_i, s in enumerate(experiment_averages[k_v]['std'].values()):
            flat_stds.append(s[0])
        flat_ds_init = []; flat_stds_init = []
        for d_i, d in enumerate(experiment_averages[k_v]['init_dist'].values()):
            flat_ds_init.append(d[0])
        for s_i, s in enumerate(experiment_averages[k_v]['init_std'].values()):
            flat_stds_init.append(s[0])

        # norm_kern = np.array(flat_ds_init)
        norm_kern = np.ones_like(flat_ds_init)
        norm_kern[np.isnan(norm_kern)] = 1.0

        bar_plot_pair_custom_labels(np.array(flat_ds_init)/norm_kern, np.array(flat_ds)/norm_kern,
                                    np.array(flat_stds_init)/norm_kern, np.array(flat_stds)/norm_kern,
                                    label_param_names, 'export', 'test',
                                    'exp_export_all_euclid_dist_params_{}.png'.format(k_v),
                                    'Avg Euclid dist per param for configuration {}'.format(k_v.replace('0_0', '0.0')).replace('_', ', '),
                                    legend=['Initial model', 'Fitted model'])

        exp_avg_ds.append(np.mean(np.array(flat_ds)/norm_kern))
        exp_avg_stds.append(np.std(np.array(flat_ds)/norm_kern))
        # exp_avg_stds.append(np.mean(flat_stds))
        exp_avg_init_ds.append(np.mean(np.array(flat_ds_init)/norm_kern))
        exp_avg_init_stds.append(np.std(np.array(flat_ds_init)/norm_kern))
        # exp_avg_init_stds.append(np.mean(flat_stds_init))

# norm_kern = 1.
# norm_kern = np.max(exp_avg_init_ds)
# bar_plot_two_grps(np.array(exp_avg_ds[:4]),
#                   np.array(exp_avg_stds[:4]),
#                   np.array(exp_avg_ds[4:]),
#                   np.array(exp_avg_stds[4:]),
#                   labels, 'export', 'test',
#                   'exp_export_all_euclid_dist_params_across_exp.eps',
#                   'Avg Euclid dist for all parameters across experiments',
#                   baseline=1.0)
bar_plot_pair_custom_labels(np.array(exp_avg_init_ds), np.array(exp_avg_ds),
                            np.array(exp_avg_init_stds), np.array(exp_avg_stds),
                            labels, 'export', 'test',
                            'exp_export_all_euclid_dist_params_across_exp_{}.png'.format(optim_to_include),
                            'Avg Euclid dist for all parameters across experiments',
                            legend=['Initial model', 'Fitted model'], baseline=1.0)

