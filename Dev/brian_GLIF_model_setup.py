from brian2 import *

start_scope()
tau = 1*ms  # "Fixes" units

GLIF_eqs = '''
dv/dt = ((G * (E_L - v) + R_I * (I_ext + I_syn_tot)) / C_m)/tau : volt
dtheta_v/dt = (a_v * (v - E_L) - b_v * (theta_v - theta_inf))/tau : volt
dtheta_s/dt = (- b_s * theta_s)/tau : volt
I_syn_tot : amp
I_ext : amp
'''

N = 3
# Parameters
G = 0.7
E_L = -65. * mV
R_I = 1. * ohm
C_m = 1.
b_s = 0.3
b_v = 0.5
a_v = 0.5
delta_theta_s = 30. * mV
f_v = 0.15
delta_V = 12. * mV
theta_inf = -20. * mV
f_I = 0.3
I_A = 100. / N * mA

reset = '''
v = E_L + f_v * (v - E_L) - delta_V
theta_s = theta_s - b_s * theta_s + delta_theta_s
'''

neurons = NeuronGroup(N=N, model=GLIF_eqs, threshold='v>30*mV', reset=reset, method='euler')
neurons.v = -65. * mV
neurons.theta_v = 1 * mV
neurons.theta_s = 30. * mV

synapses = '''
dI_syn/dt = -f_I * I_syn/tau : amp (clock-driven)
I_syn_tot_post = w * I_syn : amp (summed)
w : 1
'''
w_mean = 0.3; w_var = 0.5
rand_ws = (w_mean - w_var) + 2 * w_var * np.random.random((N, N))
S = Synapses(neurons, neurons, model=synapses, on_pre='I_syn = I_syn - f_I * I_syn + I_A', method='euler')
S.connect()
S.I_syn = 0 * mA
S.w = np.reshape(rand_ws, (-1,))
print('S.w', S.w)
