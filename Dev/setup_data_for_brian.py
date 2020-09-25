import torch

import data_util

target_data_path = data_util.prefix + data_util.path

output_fname = 'generated_spike_train_random_glif_model_t_300s_rate_0_6.mat'
output_data_path = target_data_path + output_fname
input_data_path = target_data_path + 'poisson_inputs_random_glif_model_t_300s_rate_0_6.mat'


time_interval = 4000
# time_interval = 60000
in_node_indices, input_times, input_indices = data_util.load_sparse_data(input_data_path)
spike_node_indices, spike_times, spike_indices = data_util.load_sparse_data(output_data_path)
next_target_index_list = 0
next_target_index = 0

target_params_dict = torch.load(target_data_path + 'generated_spike_train_random_glif_model_t_300s_rate_0_6_params.pt')
target_parameters = {}
for param_i, param in enumerate(target_params_dict.values()):
    target_parameters[param_i] = [param.clone().detach().numpy()]
print('target_params_dict:', target_params_dict)
