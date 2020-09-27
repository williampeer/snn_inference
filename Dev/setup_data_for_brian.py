import torch

import data_util

target_data_path = data_util.prefix + data_util.path

output_fnames_rate_0_6 = ['generated_spike_train_random_glif_1_model_t_300s_rate_0_6.mat',
                          'generated_spike_train_random_glif_2_model_t_300s_rate_0_6.mat',
                          'generated_spike_train_random_glif_3_model_t_300s_rate_0_6.mat',
                          'generated_spike_train_random_glif_slower_more_synchronous_model_t_300s_rate_0_6.mat',
                          'generated_spike_train_random_glif_slower_more_synchronous_model_t_300s_rate_0_6.mat']
output_fnames_rate_0_4 = []
target_params_rate_0_6 = []
target_params_rate_0_4 = []
for fn in output_fnames_rate_0_6:
    output_fnames_rate_0_4.append(fn.replace('_6', '_4'))
    target_params_rate_0_6.append(fn.replace('.mat', '_params.pt'))
    target_params_rate_0_4.append(fn.replace('_6.mat', '_4_params.pt'))


model_num = 0
output_fname = output_fnames_rate_0_6[model_num]
output_data_path = target_data_path + output_fname
# input_data_path = target_data_path + 'poisson_inputs_random_glif_model_t_300s_rate_0_6.mat'


time_interval = 4000
# time_interval = 60000
# in_node_indices, input_times, input_indices = data_util.load_sparse_data(input_data_path)
spike_node_indices, spike_times, spike_indices = data_util.load_sparse_data(output_data_path)
next_target_index_list = 0
next_target_index = 0

# _, sample_targets = data_util.get_spike_train_matrix(index_last_step=0, advance_by_t_steps=time_interval,
#                                                      spike_times=spike_times, spike_indices=spike_indices,
#                                                      node_numbers=spike_node_indices)

target_params_dict = torch.load(target_data_path + target_params_rate_0_6[model_num])
target_parameters = {}
index_ctr = 0
for param_i, key in enumerate(target_params_dict):
    if key not in ['loss_fn', 'rate', 'w']:
        target_parameters[index_ctr] = [target_params_dict[key].clone().detach().numpy()]
        index_ctr += 1
# print('target_parameters:', target_parameters)
