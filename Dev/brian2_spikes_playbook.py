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

N = 3
LIF_grp = NeuronGroup(N, eqs_LIF, threshold='v>30*mV', reset='v=E_L', method='euler')
# G_LIF.v = c
spikemon_LIF_grp = SpikeMonitor(LIF_grp[:], 'v', record=True)

LIF_grp.I = 100*mA

run(800*ms)

sut = spikemon_LIF_grp.spike_trains()
print('spikemon spike trains:', sut)

res = data_util.convert_brian_spike_train_dict_to_boolean_matrix(sut, t_max=800)
print('res', res)