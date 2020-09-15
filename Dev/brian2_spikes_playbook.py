from brian2 import *
from brian2modelfitting import *

import data_util

start_scope()
tau = 20*ms

# Parameters
E_L = -65*mV
R_m = 1*ohm
eqs_LIF = '''
dv/dt = (E_L - v + R_m * I)/tau : volt
I : amp
'''

LIF_grp = NeuronGroup(3, eqs_LIF, threshold='v>30*mV', reset='v=E_L', method='euler')
# G_LIF.v = c
spikemon_LIF_grp = SpikeMonitor(LIF_grp[:], 'v', record=True)

LIF_grp.I = 100*mA

run(800*ms)

sut = spikemon_LIF_grp.spike_trains()
print('spikemon spike trains:', sut)

# prefix = '/Users/william/data/target_data/'
target_data_path = data_util.prefix + data_util.path
data_path = target_data_path + 'generated_spike_train_random_glif_model_t_300s_rate_0_6.mat'
node_indices, spike_times, spike_indices = data_util.load_sparse_data(data_path)
spike_times_dict = data_util.convert_sparse_data_to_spike_times_dict(node_indices, spike_times, spike_indices)
print('spike times from loaded data:', spike_times_dict)
