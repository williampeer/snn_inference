import pymuvr
import numpy as np

from data_util import convert_to_sparse_vectors, get_spike_times_list


def get_pymuvr_dist(spike_train, targets, cos=0.2, tau=20.):
    spike_indices, spike_times = convert_to_sparse_vectors(spike_train)
    target_indices, target_times = convert_to_sparse_vectors(targets)

    t_min = np.min([spike_times[-1], target_times[-1]])
    _, spikes = get_spike_times_list(0, t_min, spike_times, spike_indices, node_numbers=np.unique(spike_indices))
    _, targets = get_spike_times_list(0, t_min, target_times, target_indices, node_numbers=np.unique(target_indices))

    for cell_i in range(0, len(spikes)):
        spikes[cell_i] = spikes[cell_i].tolist()
        targets[cell_i] = targets[cell_i].tolist()

    single_observation_dist = pymuvr.distance_matrix(trains1=[spikes], trains2=[targets], cos=cos, tau=tau)
    return single_observation_dist[0][0]
