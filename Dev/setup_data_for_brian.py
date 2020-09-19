import data_util

target_data_path = data_util.prefix + data_util.path
output_fname = 'generated_spike_train_random_glif_model_t_300s_rate_0_6.mat'
output_data_path = target_data_path + output_fname
input_data_path = target_data_path + 'poisson_inputs_random_glif_model_t_300s_rate_0_6.mat'


time_interval = 4000
# time_interval = 60000
in_node_indices, input_times, input_indices = data_util.load_sparse_data(output_data_path)
_, model_inputs = data_util.get_spike_train_matrix(index_last_step=0, advance_by_t_steps=time_interval, spike_times=input_times,
                                                   spike_indices=input_indices, node_numbers=in_node_indices)
model_inputs = model_inputs.numpy()

spike_node_indices, spike_times, spike_indices = data_util.load_sparse_data(output_data_path)
_, targets = data_util.get_spike_train_matrix(index_last_step=0, advance_by_t_steps=time_interval, spike_times=spike_times,
                                              spike_indices=spike_indices, node_numbers=spike_node_indices)
targets = targets.numpy()
