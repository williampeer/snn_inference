from brian2 import *
from brian2modelfitting import *

import data_util

start_scope()
tau = 1*ms

eqs = '''
dv/dt = ((G * (E_L - v) + R_I * (I_ext)) / C_m)/tau : volt
dtheta_v/dt = (a_v * (v - E_L) - b_v * (theta_v - theta_inf))/tau : volt
dtheta_s/dt = (- b_s * theta_s)/tau : volt
I_ext : amp
'''

# Parameters
G = 0.7
E_L = -65. * mV
# I_syn = 0. * amp
R_I = 1. * ohm
C_m = 1.
b_s = 0.3
b_v = 0.5
a_v = 0.5
delta_theta_s = 30. * mV
f_v = 0.15
delta_V = 12. * mV
f_I = 0.3
theta_inf = -20. * mV

reset = '''
v = E_L + f_v * (v - E_L) - delta_V
theta_s = theta_s - b_s * theta_s + delta_theta_s
'''

# I_syn = I_syn - f_I * I_syn + I_A

GLIF_grp = NeuronGroup(1, eqs, threshold='v>30*mV', reset=reset, method='euler')
GLIF_grp.v = -65. * mV
GLIF_grp.theta_v = 1 * mV
GLIF_grp.theta_s = 30. * mV


spikemon_LIF_grp = SpikeMonitor(GLIF_grp[:], 'v', record=True)

GLIF_grp.I_ext = 100 * mA

run(800*ms)

sut = spikemon_LIF_grp.spike_trains()
print('spikemon spike trains:', sut)

