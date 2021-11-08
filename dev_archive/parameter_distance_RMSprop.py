import os

from Models.LIF import LIF
from TargetModels.TargetModels import *
from parameter_distance import get_init_params, euclid_dist
from plot import bar_plot_pair_custom_labels

class_lookup = {'LIF': LIF, 'LIF_R': LIF_R, 'LIF_ASC': LIF_ASC, 'LIF_R_ASC': LIF_R_ASC, 'GLIF': GLIF}
target_fn_lookup = {'LIF': lif_continuous_ensembles_model_dales_compliant,
                    'LIF_R': lif_r_continuous_ensembles_model_dales_compliant,
                    'LIF_ASC': lif_asc_continuous_ensembles_model_dales_compliant,
                    'LIF_R_ASC': lif_r_asc_continuous_ensembles_model_dales_compliant,
                    'GLIF': glif_continuous_ensembles_model_dales_compliant}

# all_exps_path = '/home/william/repos/archives_snn_inference/archive_0908/archive/saved/plot_data/'
# all_exps_path = '/home/william/repos/archives_snn_inference/archive_1108_full_some_diverged/archive/saved/plot_data/'
all_exps_path = '/home/william/repos/archives_snn_inference/archive_1208_GLIF_3_LIF_R_AND_ASC_10_PLUSPLUS/archive/saved/plot_data/'
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

    param_files = []; optimiser = None; model_type = ''; lfn = 'Unknown'; spf = None
    test_losses = []
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
            test_losses.append([f_data['plot_data']['test_loss']])

    if optimiser == optim_to_include and model_type in ['LIF_R', 'LIF_R_ASC', 'GLIF'] and spf == 'None' and len(param_files) == 1:
        print('Succes! Processing exp: {}'.format(exp_folder + '/' + param_files[0]))
        exp_data = torch.load(full_folder_path + param_files[0])
        param_names = class_lookup[model_type].parameter_names
        m_p_by_exp = exp_data['plot_data']['param_means']
        if(len(m_p_by_exp)>0):
            model_N = m_p_by_exp[1][0][0].shape[0]

            config = '{}_{}_{}_{}'.format(model_type, optimiser, lfn, lr)
            if not experiment_averages.__contains__(config):
                experiment_averages[config] = { 'dist' : {}, 'std': {}, 'init_dist': {}, 'init_std': {}, 'converged': {} }
                for k in range(len(m_p_by_exp)):
                    experiment_averages[config]['dist'][param_names[k]] = []
                    experiment_averages[config]['std'][param_names[k]] = []
                    experiment_averages[config]['init_dist'][param_names[k]] = []
                    experiment_averages[config]['init_std'][param_names[k]] = []
                    experiment_averages[config]['converged'][param_names[k]] = []

            for p_i in range(len(m_p_by_exp)):
                per_exp = []
                for e_i in range(len(m_p_by_exp[p_i])):
                    init_model_params = get_init_params(class_lookup[model_type], e_i, N=model_N)
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

                converged = (test_losses[p_i][-1] + test_losses[p_i][-2]) < 0.8 * (test_losses[p_i][0]+test_losses[p_i][1])
                experiment_averages[config]['converged'][param_names[p_i]].append(converged)


# unpack
exp_avg_ds = []; exp_avg_stds = []; exp_avg_init_ds = []; exp_avg_init_stds = []
exp_converged_avg_ds = []; exp_converged_avg_stds = []; exp_converged_avg_init_ds = []; exp_converged_avg_init_stds = []
keys_list = list(experiment_averages.keys())
keys_list.sort()
labels = []
for k_i, k_v in enumerate(keys_list):
    model_type = k_v.split('_{}'.format(optim_to_include))[0]
    param_names = class_lookup[model_type].parameter_names
    label_param_names = map(lambda x: '${}$'.format(x.replace('delta_theta_', '\delta\\theta_').replace('delta_V', '\delta_V').replace('tau', '\\tau')), param_names)
    if True:
        labels.append(k_v
                      .replace('LIF_RMSprop_', 'LIF\n')
                      .replace('LIF_R_RMSprop_', 'R\n')
                      .replace('LIF_ASC_RMSprop_', 'A\n')
                      .replace('LIF_R_ASC_RMSprop_', 'R_A\n')
                      .replace('GLIF_RMSprop_', 'GLIF\n')
                      # .replace('_', '\n')
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

        flat_ds_converged = []; flat_stds_converged = []
        flat_ds_init_converged = []; flat_stds_init_converged = []
        for c_i, converged in enumerate(experiment_averages[k_v]['converged'].values()):
            if converged:
                flat_ds_converged.append(list(experiment_averages[k_v]['dist'].values())[c_i])
                flat_stds_converged.append(list(experiment_averages[k_v]['std'].values())[c_i])
                flat_ds_init_converged.append(list(experiment_averages[k_v]['init_dist'].values())[c_i])
                flat_stds_init_converged.append(list(experiment_averages[k_v]['init_std'].values())[c_i])

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
        exp_avg_init_ds.append(np.mean(np.array(flat_ds_init)/norm_kern))
        exp_avg_init_stds.append(np.std(np.array(flat_ds_init)/norm_kern))

        bar_plot_pair_custom_labels(np.array(flat_ds_converged) / norm_kern, np.array(flat_ds_init_converged) / norm_kern,
                                    np.array(flat_stds_converged) / norm_kern, np.array(flat_stds_init_converged) / norm_kern,
                                    label_param_names, 'export', 'test',
                                    'exp_export_converged_euclid_dist_params_{}.png'.format(k_v),
                                    'Avg Euclid dist per param for configuration {}'.format(
                                        k_v.replace('0_0', '0.0')).replace('_', ', '),
                                    legend=['Fitted model', 'Initial model'])
        exp_converged_avg_ds.append(np.mean(np.array(flat_ds_converged)/norm_kern))
        exp_converged_avg_stds.append(np.std(np.array(flat_ds_converged)/norm_kern))
        exp_converged_avg_init_ds.append(np.mean(np.array(flat_ds_init_converged)/norm_kern))
        exp_converged_avg_init_stds.append(np.std(np.array(flat_ds_init_converged)/norm_kern))

bar_plot_pair_custom_labels(np.array(exp_avg_ds), np.array(exp_avg_init_ds),
                            np.array(exp_avg_stds), np.array(exp_avg_init_stds),
                            labels, 'export', 'test',
                            'exp_export_all_euclid_dist_params_across_exp_{}.png'.format(optim_to_include),
                            'Avg Euclid dist for all parameters across experiments',
                            legend=['Fitted model', 'Initial model'], baseline=1.0)

bar_plot_pair_custom_labels(np.array(exp_converged_avg_ds), np.array(exp_converged_avg_init_ds),
                            np.array(exp_converged_avg_stds), np.array(exp_converged_avg_init_stds),
                            labels, 'export', 'test',
                            'exp_export_all_euclid_dist_params_across_exp_converged_loss_{}.png'.format(optim_to_include),
                            'Avg Euclid dist for all parameters across experiments',
                            legend=['Fitted model', 'Initial model'], baseline=1.0)

