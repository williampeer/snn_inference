import numpy as np
import torch

from data_util import save_spiketrain_in_matlab_format, convert_to_sparse_vectors
from experiments import generate_synthetic_data
from plot import plot_spiketrain

# path = './Test/LIF_test.pt'
# path = './Test/IzhikevichStable_test.pt'
path = './Test/IzhikevichStable_sample.pt'
model = torch.load(path)['model']

t = 30 * 5 * 1000
interval_size = 4000
interval_range = int(t/interval_size)
poisson_rate = 0.6
# rate_wake = 0.6
# rate_rem = 0.6
# rate_nrem = 0.3

spike_indices = np.array([], dtype='int8')
spike_times = np.array([], dtype='float32')
for t_i in range(interval_range):  # 1s at a time
    model.reset_hidden_state()
    spiketrain = generate_synthetic_data(model, poisson_rate, t=interval_size)
    plot_spiketrain(spiketrain, 'Plot imported neuron', 'test_import_neuron')
    cur_spike_indices, cur_spike_times = convert_to_sparse_vectors(spiketrain)
    spike_indices = np.append(spike_indices, cur_spike_indices)
    spike_times = np.append(spike_times, cur_spike_times)

save_spiketrain_in_matlab_format(fname='test_model_spiketrain.mat', spike_indices=spike_indices, spike_times=spike_times)
