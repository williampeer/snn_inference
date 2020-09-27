import scipy.io as sio
import numpy as np
import torch
import brian2

# NOTE: This is an implementation for sparse representations (two vectors) of a spike trains,
#   represented by two vectors; the spike times, and node indices.

# prefix = '/home/william/'  # Ubuntu
# prefix = '/Users/william/'  # OS X
prefix = '/home/williampeer/'  # server
path = 'data/target_data/'
matlab_export = 'matlab_export/'


def load_sparse_data(full_path):
    exp_data = sio.loadmat(full_path)['DATA']

    spike_indices = exp_data['clu'][0][0]  # index of the spiking neurons
    spike_times = exp_data['res'][0][0]  # spike times

    node_indices = np.unique(spike_indices)

    return node_indices, spike_times, spike_indices


def convert_to_sparse_vectors(spiketrain, t_offset):
    assert spiketrain.shape[0] > spiketrain.shape[1], "assuming bins x nodes (rows as timesteps). spiketrain.shape: {}".format(spiketrain.shape)

    spike_indices = np.array([], dtype='int8')
    spike_times = np.array([], dtype='float32')
    for ms_i in range(spiketrain.shape[0]):
        for node_i in range(spiketrain.shape[1]):
            if spiketrain[ms_i][node_i] != 0:
                assert spiketrain[ms_i][node_i] in range(10), \
                    "element out of range(0,5). row: {}, col: {}, value:{}".format(ms_i, node_i, spiketrain[ms_i][node_i])
                # assert spiketrain[ms_i][node_i] == 1, \
                #     "found element that was neither 0 nor 1. row: {}, col: {}, value:{}".format(ms_i, node_i, spiketrain[ms_i][node_i])
                spike_times = np.append(spike_times, np.float32(float(ms_i) +t_offset))
                spike_indices = np.append(spike_indices, np.int8(node_i))

    return spike_indices, spike_times


def save_spiketrain_in_sparse_matlab_format(fname, spike_indices, spike_times):
    exp_data = {}
    exp_data['clu'] = np.reshape(spike_indices, (-1, 1))
    exp_data['res'] = np.reshape(spike_times, (-1, 1))
    mat_data = {'DATA': exp_data}

    # sio.savemat(file_name=prefix + path + matlab_export + fname, mdict=mat_data)
    # sio.savemat(file_name='/Users/william/repos/pnmf-fork/data/' + fname, mdict=mat_data)
    sio.savemat(file_name=prefix + path + fname, mdict=mat_data)


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


def transform_states(states, bin_size, target_bin_size):
    res_states = torch.zeros((states.shape[0] * int(bin_size/target_bin_size), states.shape[1]))
    for states_i in range(states.shape[0]):
        for expand_i in range(int(bin_size/target_bin_size)):
            res_states[states_i+expand_i] = torch.from_numpy(states[states_i])

    return res_states


def sample_state_input(state_labels_1d, n_dim):
    # state labels as state 0, 1, or 2
    broadcast_state_labels = state_labels_1d + torch.zeros((state_labels_1d.shape[0], n_dim))

    # NREM is slower, more synchronous, REM and wake quicker, asynchronous (EEG)
    # --> REM/wake: higher row-p, lower col. p
    # --> NREM:     lower row-p, higher/medium col. p ?
    wake_rem_state = 1.0 * (broadcast_state_labels != 2)
    wake_rem_mask_row = (torch.rand(broadcast_state_labels.shape) > 0.25)
    wake_rem_mask_col = (torch.rand(broadcast_state_labels.shape) > 0.8)
    wake_rem_mask = wake_rem_mask_row * wake_rem_mask_col

    nrem_state = 1.0 * (broadcast_state_labels == 2)
    nrem_mask_row = (torch.rand(broadcast_state_labels.shape) < 0.8)
    nrem_mask_col = (torch.rand(broadcast_state_labels.shape) < 0.25)
    nrem_mask = nrem_mask_row * nrem_mask_col

    sample_input = wake_rem_state * wake_rem_mask + nrem_state * nrem_mask
    return sample_input.float()


def get_sample_inputs_from_states(states, train_i, states_per_train_iter, bin_size, target_bin_size, N):
    cur_states = states[train_i * states_per_train_iter:(train_i + 1) * states_per_train_iter]
    unwrap_states = transform_states(cur_states, bin_size=bin_size, target_bin_size=target_bin_size)
    return sample_state_input(unwrap_states, n_dim=N)


def transform_to_population_spiking(spikes, kernel_indices):
    convolved_spikes = torch.zeros((spikes.shape[0], len(kernel_indices)))
    for t_i in range(spikes.shape[0]):
        for pop_i in range(len(kernel_indices)):
            for idx in range(len(kernel_indices[pop_i])):
                convolved_spikes[t_i, pop_i] += spikes[t_i, kernel_indices[pop_i][idx]]

    return torch.min(convolved_spikes, torch.tensor(1.0))


# def load_sparse_data_matlab_format(fname):
#     exp_data = sio.loadmat(prefix + path + fname)['DATA']
#
#     # Custom Matlab-compatible format
#     spike_indices = exp_data['clu'][0][0]  # index of the spiking neurons
#     spike_times = exp_data['res'][0][0]  # spike times
#     qual = exp_data['qual'][0][0]  # neuronal decoding quality
#     states = exp_data['score'][0][0]  # state
#
#     satisfactory_quality_node_indices = np.unique(spike_indices)
#
#     return satisfactory_quality_node_indices, spike_times, spike_indices, qual, states


def convert_brian_spike_train_dict_to_boolean_matrix(brian_spike_train, t_max):
    keys = brian_spike_train.keys()
    res = np.zeros((int(t_max), len(keys)))
    for i, k in enumerate(keys):
        node_spike_times = brian_spike_train[k]
        node_spike_times = np.array(node_spike_times/brian2.msecond, dtype=np.int)
        res[node_spike_times, i] = 1.
    return res


def convert_brian_spike_train_to_matlab_format(brian_spike_train):
    spike_indices = np.array([], dtype='int8')
    spike_times = np.array([], dtype='float32')

    pass


def convert_sparse_spike_train_to_matrix(spike_times, node_indices, unique_node_indices):
    res = {}
    for j in range(len(unique_node_indices)):
        res[int(unique_node_indices[j])] = np.array([])
    for i in range(len(spike_times)):
        res[int(node_indices[i])] = np.concatenate((res[int(node_indices[i])], np.array(spike_times[i])))
    return res


def get_spike_times_list(index_last_step, advance_by_t_steps, spike_times, spike_indices, node_numbers):
    res = []
    for _ in range(len(node_numbers)):
        res.append([])

    prev_spike_time = spike_times[index_last_step]

    next_step = index_last_step+1
    while spike_times[next_step] < prev_spike_time + advance_by_t_steps:
        cur_node_index = int(spike_indices[next_step])
        res[cur_node_index] = np.concatenate((res[cur_node_index], spike_times[next_step]))
        next_step = next_step+1

    return next_step, res


def scale_spike_times(spike_times_list, div=1000.):
    for n_i in range(len(spike_times_list)):
        spike_times_list[n_i] = spike_times_list[n_i] / div
    return  spike_times_list
