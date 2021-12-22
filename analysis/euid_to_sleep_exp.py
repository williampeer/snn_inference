import os

import numpy as np
import analysis_util

experiments_path = '/media/william/p6/archive_14122021/archive/saved/sleep_data_no_types/'
# experiments_path_plot_data = '/media/william/p6/archive_14122021/archive/saved/plot_data/sleep_data_no_types/'

experiments_path_sleep_data_microGIF = '/home/william/repos/snn_inference/Test/saved/sleep_data/'
# experiments_path_plot_sleep_data_microGIF = '/home/william/repos/snn_inference/Test/saved/plot_data/sleep_data/'

sleep_exps = ['exp108', 'exp109', 'exp124', 'exp126', 'exp138', 'exp146', 'exp147']
sleep_data_approx_rates = []
for sleep_exp in sleep_exps:
    sleep_data_rate = analysis_util.get_target_rate_for_sleep_exp(sleep_exp)
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
