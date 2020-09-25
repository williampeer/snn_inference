import torch

from brian2 import *
from Dev.setup_data_for_brian import *
from Log import Logger
from eval import calculate_loss
from gf_metric import compute_gamma_factor_for_lists, get_spikes

logger = Logger(log_fname='brian2_network_nevergrad_optimization')
N = 12; tau_vr = 4.0

GLIF_eqs = '''
dv/dt = ((G * (E_L - v) + R_I * (I_ext + I_syn_tot)) / C_m)/tau : volt
dtheta_v/dt = (a_v * (v - E_L) - b_v * (theta_v - theta_inf))/tau : volt
dtheta_s/dt = (- b_s * theta_s)/tau : volt
I_syn_tot : amp
I_ext : amp
E_L : volt
R_I : ohm
G : 1
f_v : 1
C_m : 1
f_I : 1
b_s : 1
b_v : 1
a_v : 1
delta_theta_s : volt
delta_V : volt
theta_inf : volt
I_A : amp
'''

reset = '''
v = E_L + f_v * (v - E_L) - delta_V
theta_s = theta_s - b_s * theta_s + delta_theta_s
'''

in_eqs = '''
dI_in/dt = -I_in/tau : amp (clock-driven)
I_ext_post = I_in : amp (summed)
'''

synapse_eqs = '''
dI_syn/dt = -f_I * I_syn/tau : amp (clock-driven)
I_syn_tot_post = w * I_syn : amp (summed)
w : 1
'''


def run_simulation_multiobjective(w, C_m, G, R_I, f_v, f_I, E_L, b_s, b_v, a_v, delta_theta_s, delta_V, theta_inf, I_A, t_interval=time_interval*ms):
    start_scope()
    tau = 1*ms

    neurons = NeuronGroup(N=N, model=GLIF_eqs, threshold='v>30*mV', reset=reset, method='euler')
    neurons.set_states({'f_I': f_I, 'C_m': C_m, 'G': G, 'R_I': R_I*ohm, 'f_v': f_v, 'E_L': E_L*mV, 'b_s': b_s, 'b_v': b_v,
                        'a_v': a_v, 'delta_theta_s': delta_theta_s * mV, 'delta_V': delta_V * mV, 'theta_inf': theta_inf * mV, 'I_A': I_A*mA})

    in_grp = SpikeGeneratorGroup(N, np.reshape(input_indices, (-1,)), np.reshape(input_times*ms, (-1,)))
    feedforward = Synapses(in_grp, neurons, model=in_eqs, on_pre='I_in = 1 * mA')
    feedforward.connect(j='i')
    # in_grp.set_spikes(np.reshape(input_indices, (-1,)), np.reshape(input_times*ms, (-1,)))

    synapses = Synapses(neurons, neurons, model=synapse_eqs, on_pre='I_syn = I_syn - f_I * I_syn + I_A', method='euler')
    synapses.connect()
    synapses.set_states({'w': w})

    spikemon = SpikeMonitor(neurons[:], 'v', record=True)

    run(t_interval)
    silent_model = spikemon.num_spikes == 0
    if silent_model:
        logger.log("------------- WARN: no spikes in spikes observed")
        return [12 * 10 * neurons.v.shape[0], 20., 20.]

    rand_sample_index = int(0.6 * np.random.rand() * spike_times.shape[0])
    m_spike_times = get_spikes(spikemon)
    _, target_spike_times = data_util.get_spike_times_list(index_last_step=rand_sample_index,
                                                           advance_by_t_steps=time_interval, spike_times=spike_times,
                                                           spike_indices=spike_indices, node_numbers=spike_node_indices)
    t_spike_times = data_util.scale_spike_times(target_spike_times)  # ms to seconds
    gf = compute_gamma_factor_for_lists(m_spike_times, t_spike_times, time=t_interval, delta=1*ms)

    brian_model_spike_train = data_util.convert_brian_spike_train_dict_to_boolean_matrix(spikemon.spike_trains(), t_max=t_interval/ms)
    brian_model_spike_train = torch.tensor(brian_model_spike_train, dtype=torch.float)
    _, targets = data_util.get_spike_train_matrix(index_last_step=rand_sample_index,
                                                  advance_by_t_steps=time_interval, spike_times=spike_times,
                                                  spike_indices=spike_indices, node_numbers=spike_node_indices)
    vr_dist = np.float(calculate_loss(brian_model_spike_train, targets, loss_fn='van_rossum_dist', tau_vr=tau_vr))
    poisson_nll = np.float(calculate_loss(brian_model_spike_train, targets, loss_fn='poisson_nll'))

    logger.log('current losses, rand_sample_index: {}'.format(rand_sample_index), parameters=[vr_dist, poisson_nll, gf])
    return [vr_dist, poisson_nll, gf]


def run_simulation_for(rate, w, C_m, G, R_I, f_v, f_I, E_L, b_s, b_v, a_v, delta_theta_s, delta_V, theta_inf, I_A, loss_fn, t_interval=4000*ms):
    # restore('init')
    start_scope()
    tau = 1 * ms

    neurons = NeuronGroup(N=N, model=GLIF_eqs, threshold='v>30*mV', reset=reset, method='euler')
    neurons.set_states(
        {'f_I': f_I, 'C_m': C_m, 'G': G, 'R_I': R_I * ohm, 'f_v': f_v, 'E_L': E_L * mV, 'b_s': b_s, 'b_v': b_v,
         'a_v': a_v, 'delta_theta_s': delta_theta_s * mV, 'delta_V': delta_V * mV, 'theta_inf': theta_inf * mV,
         'I_A': I_A * mA})

    in_grp = SpikeGeneratorGroup(N, np.reshape(input_indices, (-1,)), np.reshape(input_times * ms, (-1,)))
    feedforward = Synapses(in_grp, neurons, model=in_eqs, on_pre='I_in = 1 * mA')
    feedforward.connect(j='i')
    # in_grp.set_spikes(np.reshape(input_indices, (-1,)), np.reshape(input_times*ms, (-1,)))

    synapses = Synapses(neurons, neurons, model=synapse_eqs, on_pre='I_syn = I_syn - f_I * I_syn + I_A', method='euler')
    synapses.connect()
    synapses.set_states({'w': w})

    spikemon = SpikeMonitor(neurons[:], 'v', record=True)

    run(t_interval)
    if spikemon.num_spikes == 0:
        logger.log("------------- WARN: no spikes in spikes observed")
        return np.inf

    if loss_fn == 'gamma_factor':
        m_spike_times = get_spikes(spikemon)
        _, target_spike_times = data_util.get_spike_times_list(
            index_last_step=int(0.6 * np.random.rand() * spike_times.shape[0]),
            advance_by_t_steps=time_interval, spike_times=spike_times,
            spike_indices=spike_indices, node_numbers=spike_node_indices)
        t_spike_times = data_util.scale_spike_times(target_spike_times)  # ms to seconds
        loss = compute_gamma_factor_for_lists(m_spike_times, t_spike_times, time=t_interval, delta=1 * ms)
        logger.log('loss_fn: gamma_factor, loss: {:3.3f}'.format(loss))
        return loss
    elif loss_fn in ['van_rossum_dist', 'poisson_nll', 'kl_div']:
        brian_model_spike_train = data_util.convert_brian_spike_train_dict_to_boolean_matrix(spikemon.spike_trains(),
                                                                                             t_max=t_interval / ms)
        brian_model_spike_train = torch.tensor(brian_model_spike_train, dtype=torch.float)
        _, targets = data_util.get_spike_train_matrix(
            index_last_step=int(0.6 * np.random.rand() * spike_times.shape[0]),
            advance_by_t_steps=time_interval, spike_times=spike_times,
            spike_indices=spike_indices, node_numbers=spike_node_indices)
        loss = np.float(calculate_loss(brian_model_spike_train, targets, loss_fn=loss_fn, tau_vr=tau_vr))
        logger.log('loss_fn: {}, loss: {:3.3f}'.format(loss_fn, loss))
        return loss
