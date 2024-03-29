import torch

import data_util
from Dev.brian2_custom_network_opt import get_spike_train_for_matlab_export
from data_util import save_spiketrain_in_sparse_matlab_format, convert_brian_spike_train_to_matlab_format
from experiments import zip_dicts

# dict_path = '/home/william/repos/archives_snn_inference/archive (4)/saved/single_objective_optim/fitted_params_optim_CMA_loss_fn_vrdfrd_budget_10000_exp_0.pt'
# dict_path = '/home/william/repos/archives_snn_inference/archive (4)/saved/single_objective_optim/fitted_params_optim_CMA_loss_fn_vrdfrd_budget_10000_exp_1.pt'
# dict_path = '/home/william/repos/archives_snn_inference/archive (4)/saved/single_objective_optim/fitted_params_optim_CMA_loss_fn_vrdfrd_budget_10000_exp_2.pt'
# dict_path = '/home/william/repos/archives_snn_inference/archive (4)/saved/single_objective_optim/fitted_params_optim_DE_loss_fn_vrdfrd_budget_10000_exp_0.pt'
# dict_path = '/home/william/repos/archives_snn_inference/archive (4)/saved/single_objective_optim/fitted_params_optim_DE_loss_fn_vrdfrd_budget_10000_exp_1.pt'
dict_path = '/home/william/repos/archives_snn_inference/archive (4)/saved/single_objective_optim/fitted_params_optim_DE_loss_fn_vrdfrd_budget_10000_exp_2.pt'

optim_name = 'CMA'
model_parameters = torch.load(dict_path)
print('Loaded models dict.')


# def convert_integer_indexed_to_named_params(d):
#     parameter_names = ['E_L', 'C_m', 'G', 'R_I', 'f_v', 'f_I', 'delta_theta_s', 'b_s', 'a_v', 'b_v', 'theta_inf', 'delta_V', 'I_A']
#     named_params_dict = {}
#     for i in range(13):
#         named_params_dict[parameter_names[i]] = d[i]
#     return named_params_dict


# model_parameters = convert_integer_indexed_to_named_params(params_by_optim)
# for exp_i in range(len(model_parameters['E_L'])):
weights = model_parameters['w']
rate = model_parameters['rate']

current_model_parameters = {}
for i, k in enumerate(model_parameters):  # TODO: fix
    if k not in ['w', 'rate', 'loss_fn']:
        current_model_parameters[k] = model_parameters[k]

brian_spikes = get_spike_train_for_matlab_export(rate, weights, current_model_parameters)  #, run_time=4000)  # default 5m.
spike_times, node_spike_indices = convert_brian_spike_train_to_matlab_format(brian_spikes)

fname = dict_path.split('/')[-1]  # TODO: double check
model_name = fname.split('.pt')[0]

save_fname_output = 'spikes_brian_{}_rate_{:2.2f}_exp_{}'.format(model_name, rate, dict_path.split('exp_')[-1].split('.pt')[0]).replace('.', '_') + '.mat'
save_spiketrain_in_sparse_matlab_format(fname=save_fname_output, spike_indices=node_spike_indices, spike_times=spike_times)

combined_model_params = {'w': weights, 'rate': rate}
combined_model_params = zip_dicts(combined_model_params, current_model_parameters)
full_save_path = data_util.prefix + data_util.target_data_path + fname
torch.save(combined_model_params, full_save_path.replace('.mat', '_params.pt'))
