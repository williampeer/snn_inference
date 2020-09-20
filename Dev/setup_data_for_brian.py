import data_util
from gf_metric import get_gamma_factor, get_spikes

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


_, sut = data_util.get_spike_times_list(index_last_step=0, advance_by_t_steps=time_interval, spike_times=spike_times,
                                        spike_indices=spike_indices, node_numbers=spike_node_indices)
_, sut2 = data_util.get_spike_times_list(index_last_step=0, advance_by_t_steps=time_interval, spike_times=input_times,
                                         spike_indices=input_indices, node_numbers=in_node_indices)
model_spike_times = data_util.scale_spike_times(sut)
target_spike_times = data_util.scale_spike_times(sut2)

import gf_metric
from brian2 import *
gf = gf_metric.compute_gamma_factor_for_lists(model_spike_times, target_spike_times,
                                              delta=2*ms, time=float(time_interval)*ms, dt_=1*ms)
print('gf', gf)
