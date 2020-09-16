import brian2modelfitting as b2f
import data_util

from brian2 import *

# from Dev.brian_GLIF_model_setup import *

# spikemon_LIF_grp = SpikeMonitor(neurons[:], 'v', record=True)
# neurons.I_ext = 80 * mA
# run(100*ms)
# spikemon = spikemon_LIF_grp.spike_trains()
# print('spikemon spike trains:', spikemon)

target_data_path = data_util.prefix + data_util.path
output_data_path = target_data_path + 'generated_spike_train_random_glif_model_t_300s_rate_0_6.mat'
input_data_path = target_data_path + 'poisson_inputs_random_glif_model_t_300s_rate_0_6.mat'

t = 4000
in_node_indices, input_times, input_indices = data_util.load_sparse_data(output_data_path)
# input_spike_times = data_util.convert_sparse_data_to_spike_times_dict(in_node_indices, input_times, input_indices)
_, first_inputs = data_util.get_spike_train_matrix(index_last_step=0, advance_by_t_steps=t, spike_times=input_times,
                                                   spike_indices=input_indices, node_numbers=in_node_indices)
first_inputs = first_inputs.numpy() * 100 * mA

spike_node_indices, spike_times, spike_indices = data_util.load_sparse_data(output_data_path)
# node_spike_times = data_util.convert_sparse_data_to_spike_times_dict(spike_node_indices, spike_times, spike_indices)
_, first_outputs = data_util.get_spike_train_matrix(index_last_step=0, advance_by_t_steps=t, spike_times=spike_times,
                                                    spike_indices=spike_indices, node_numbers=spike_node_indices)

GLIF_eqs = '''
dv/dt = ((G * (E_L - v) + R_I * I) / C_m)/tau : volt
dtheta_v/dt = (a_v * (v - E_L) - b_v * (theta_v - theta_inf))/tau : volt
dtheta_s/dt = (- b_s * theta_s)/tau : volt
C_m : 1 (constant)
G : 1 (constant)
R_I : ohm (constant)
f_v : 1 (constant)
b_s : 1
b_v : 1
a_v : 1
E_L : volt
delta_theta_s : volt
delta_V : volt
theta_inf : volt
'''
reset = '''
v = E_L + f_v * (v - E_L) - delta_V
theta_s = theta_s - b_s * theta_s + delta_theta_s
'''

init_params = {'C_m': 1.5, 'G': 0.8, 'R_I': 18., 'E_L': -60.,
               'delta_theta_s': 25., 'b_s': 0.4, 'f_v': 0.14, 'delta_V': 12.,
               'b_v': 0.5, 'a_v': 0.5, 'theta_inf': -25.}
fitter = b2f.fitter.SpikeFitter(GLIF_eqs, input=first_inputs, output=first_outputs, dt=0.5*ms, reset=reset,
                                threshold='v>30*mV', refractory=False,
                                n_samples=100, method='euler', param_init=init_params,
                                penalty=None, use_units=True)

results, error = fitter.fit(n_rounds=2, optimizer=b2f.SkoptOptimizer('ET'),
                            metric=b2f.GammaFactor(dt=0.1*ms, delta=1*ms))
