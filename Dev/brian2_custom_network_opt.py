import torch
from brian2 import *

import data_util
from Log import Logger
from eval import calculate_loss
from experiments import generate_synthetic_data

logger = Logger(log_fname='brian2_network_nevergrad_optimization')

start_scope()
tau = 1 * ms

N = 12; tau_vr = 4.0

GLIF_eqs = '''
dv/dt = (G * (E_L - v) + R_I*1/(1+exp(-(I_ext + I_syn_tot)) ) / C_m)/tau : 1
dtheta_v/dt = (a_v * (v - E_L) - b_v * (theta_v - theta_inf))/tau : 1
dtheta_s/dt = (- b_s * theta_s)/tau : 1
I_ext : 1
I_syn_tot : 1
E_L : 1
R_I : 1
G : 1
f_v : 1
C_m : 1
f_I : 1
b_s : 1
b_v : 1
a_v : 1
delta_theta_s : 1
delta_V : 1
theta_inf : 1
I_A : 1
'''

reset = '''
v = E_L + f_v * (v - E_L) - delta_V
theta_s = theta_s - b_s * theta_s + delta_theta_s
'''

in_eqs = '''
I_ext_post = I_in : 1 (summed)
dI_in/dt = -I_in/tau : 1 (clock-driven)
'''

synapse_eqs = '''
I_syn_tot_post = w * I_syn : 1 (summed)
dI_syn/dt = -f_I * I_syn/tau : 1 (clock-driven)
w : 1
'''

neurons = NeuronGroup(N=N, model=GLIF_eqs, threshold='v>(theta_s + theta_v)', reset=reset, method='euler')

synapses = Synapses(neurons, neurons, model=synapse_eqs, on_pre='I_syn = I_syn - f_I * I_syn + I_A', method='euler')
synapses.connect(condition=True)

poisson_input_grp = PoissonGroup(N, 30.*Hz)
feedforward = Synapses(poisson_input_grp, neurons, model=in_eqs, on_pre='I_in = 1', method='euler')
feedforward.connect(j='i')

spikemon = SpikeMonitor(neurons[:], 'v', record=True)

store()


def convert_brian_spike_train_dict_to_boolean_matrix(brian_spike_train, t_max):
    keys = brian_spike_train.keys()
    res = np.zeros((int(t_max), len(keys)))
    for i, k in enumerate(keys):
        node_spike_times = brian_spike_train[k]
        node_spike_times = np.array(node_spike_times/brian2.msecond, dtype=np.int)
        res[node_spike_times, i] = 1.
    return res


def convert_brian_spike_train_to_matlab_format(brian_spikes):
    spike_indices = np.array([], dtype='int8')
    spike_times = np.array([], dtype='float32')

    for n_i in brian_spikes.keys():
        spike_times_ms = np.round(brian_spikes[n_i]/brian2.ms)
        spike_times = np.concatenate((spike_times, spike_times_ms))
        spike_indices = np.concatenate((spike_indices, n_i * np.ones_like(spike_times_ms, dtype='int8')))

    indices_sorted = np.argsort(spike_times)
    return spike_times[indices_sorted], spike_indices[indices_sorted]


def run_simulation_for(rate, w, C_m, G, R_I, f_v, f_I, E_L, b_s, b_v, a_v, delta_theta_s, delta_V, theta_inf, I_A, loss_fn,
                       target_model, target_rate, time_interval=4000):
    restore()

    neurons.set_states(
        {'f_I': f_I, 'C_m': C_m, 'G': G, 'R_I': R_I, 'f_v': f_v, 'E_L': E_L, 'b_s': b_s, 'b_v': b_v, 'a_v': a_v,
         'delta_theta_s': delta_theta_s, 'delta_V': delta_V, 'theta_inf': theta_inf, 'I_A': I_A})

    print('E_L', neurons.get_states({'E_L'}))
    poisson_input_grp.rates = rate * Hz

    synapses.set_states({'w': w})

    run(time_interval*ms)
    print('DEBUG: spikemon.num_spikes: {}'.format(spikemon.num_spikes))
    if spikemon.num_spikes == 0:
        logger.log("------------- WARN: no spikes in spikes observed")
        return np.inf

    # if loss_fn == 'gamma_factor':
    #     m_spike_times = get_spikes(spikemon)
    #     _, target_spike_times = data_util.get_spike_times_list(
    #         index_last_step=int(0.7 * np.random.rand() * spike_times.shape[0]),
    #         advance_by_t_steps=int(t_interval/ms), spike_times=spike_times,
    #         spike_indices=spike_indices, node_numbers=spike_node_indices)
    #     t_spike_times = data_util.scale_spike_times(target_spike_times)  # ms to seconds
    #     loss = compute_gamma_factor_for_lists(m_spike_times, t_spike_times, time=t_interval, delta=1 * ms)
    #     logger.log('loss_fn: gamma_factor, loss: {:3.3f}'.format(loss))
    #     return loss
    elif loss_fn in ['van_rossum_dist', 'poisson_nll', 'kl_div', 'vrdfrd']:
        brian_model_spike_train = data_util.convert_brian_spike_train_dict_to_boolean_matrix(spikemon.spike_trains(),
                                                                                             t_max=time_interval)
        brian_model_spike_train = torch.tensor(brian_model_spike_train, dtype=torch.float)
        targets = generate_synthetic_data(target_model, target_rate, time_interval)
        loss = np.float(calculate_loss(brian_model_spike_train, targets, loss_fn=loss_fn, tau_vr=tau_vr))
        logger.log('loss_fn: {}, loss: {:3.3f}'.format(loss_fn, loss))
        return loss


def get_spike_train_for(rate, weights, neurons_params, run_time=4000):
    restore()

    neurons.set_states(neurons_params)

    poisson_input_grp.rates = rate * Hz

    synapses.set_states({'w': weights})

    mon_in = SpikeMonitor(poisson_input_grp[:], record=True)

    run(run_time*ms)
    print('spikes:', spikemon.num_spikes)
    print('mon_in spikes:', mon_in.num_spikes)

    return torch.tensor(data_util.convert_brian_spike_train_dict_to_boolean_matrix(spikemon.spike_trains(), t_max=run_time), dtype=torch.float32)


def get_spike_train_for_matlab_export(rate, weights, neurons_params, run_time=60*1000):
    restore()

    neurons.set_states(neurons_params)

    poisson_input_grp.rates = rate * Hz

    synapses.set_states({'w': weights})

    run(run_time * ms)
    print('spikes:', spikemon.num_spikes)

    return spikemon.spike_trains()
