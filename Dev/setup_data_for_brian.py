import data_util

target_data_path = data_util.prefix + data_util.target_data_path

output_fnames_rate_0_6 = ['generated_spike_train_random_glif_1_model_t_300s_rate_0_6.mat',
                          'generated_spike_train_random_glif_2_model_t_300s_rate_0_6.mat',
                          'generated_spike_train_random_glif_3_model_t_300s_rate_0_6.mat',
                          'generated_spike_train_glif_slower_rate_async_t_300s_rate_0_6.mat',
                          'generated_spike_train_glif_slower_more_synchronous_model_t_300s_rate_0_6.mat']
output_fnames_rate_0_4 = []
target_params_rate_0_6 = []
target_params_rate_0_4 = []
for fn in output_fnames_rate_0_6:
    output_fnames_rate_0_4.append(fn.replace('_6', '_4'))
    target_params_rate_0_6.append(fn.replace('.mat', '_params.pt'))
    target_params_rate_0_4.append(fn.replace('_6.mat', '_4_params.pt'))

all_output_fnames = output_fnames_rate_0_6
for fn in output_fnames_rate_0_4:
    all_output_fnames.append(fn)


# model_num = 4
# output_fname = output_fnames_rate_0_6[model_num]
# output_data_path = target_data_path + output_fname
# # input_data_path = target_data_path + 'poisson_inputs_random_glif_model_t_300s_rate_0_6.mat'
#
#
# time_interval = 4000
# spike_node_indices, spike_times, spike_indices = data_util.load_sparse_data(output_data_path)
# next_target_index_list = 0
# next_target_index = 0
#
# target_params_dict = torch.load(target_data_path + target_params_rate_0_6[model_num])
# target_parameters = {}
# index_ctr = 0
# for param_i, key in enumerate(target_params_dict):
#     if key not in ['loss_fn', 'rate', 'w']:
#         target_parameters[index_ctr] = [target_params_dict[key].clone().detach().numpy()]
#         index_ctr += 1
