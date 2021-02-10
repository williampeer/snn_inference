import numpy as np
import torch

prefix = '/Users/user/'
path = 'data/target_data/'
matlab_export = 'matlab_export/'


def get_spike_train_matrix(index_last_step, advance_by_t_steps, spike_times, spike_indices, node_numbers):
    spikes = torch.zeros((advance_by_t_steps, node_numbers.shape[0]))

    prev_spike_time = spike_times[index_last_step]

    next_step = index_last_step+1
    next_spike_time = spike_times[next_step]
    while next_spike_time < prev_spike_time + advance_by_t_steps:
        spike_arr_index = np.where(node_numbers == spike_indices[next_step])[0]
        time_index = int(np.floor(next_spike_time[0] - prev_spike_time[0]))
        spikes[time_index][spike_arr_index] = 1.0
        next_step = next_step+1
        next_spike_time = spike_times[next_step]

    return next_step, spikes


def transform_to_population_spiking(spikes, kernel_indices):
    convolved_spikes = torch.zeros((spikes.shape[0], len(kernel_indices)))
    for t_i in range(spikes.shape[0]):
        for pop_i in range(len(kernel_indices)):
            for idx in range(len(kernel_indices[pop_i])):
                convolved_spikes[t_i, pop_i] += spikes[t_i, kernel_indices[pop_i][idx]]

    return torch.min(convolved_spikes, torch.tensor(1.0))


def convert_sparse_spike_train_to_matrix(spike_times, node_indices, unique_node_indices):
    res = {}
    for j in range(len(unique_node_indices)):
        res[int(unique_node_indices[j])] = np.array([])
    for i in range(len(spike_times)):
        res[int(node_indices[i])] = np.concatenate((res[int(node_indices[i])], np.array(spike_times[i])))
    return res


def get_spike_times_list(index_last_step, advance_by_t_steps, spike_times, spike_indices, num_nodes):
    res = []
    for _ in range(num_nodes):
        res.append([])

    if index_last_step == 0:
        prev_spike_time = 0
    else:
        prev_spike_time = spike_times[index_last_step]

    next_step = index_last_step+1
    while next_step < len(spike_times) and spike_times[next_step] < prev_spike_time + advance_by_t_steps:
        cur_node_index = int(spike_indices[next_step])
        res[cur_node_index] = np.concatenate((res[cur_node_index], [spike_times[next_step]]))
        next_step = next_step+1

    return next_step, res


def scale_spike_times(spike_times_list, div=1000.):
    for n_i in range(len(spike_times_list)):
        spike_times_list[n_i] = spike_times_list[n_i] / div
    return  spike_times_list
