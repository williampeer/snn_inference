from brian2 import *

from Dev.setup_data_for_brian import *
from Log import Logger
from eval import calculate_loss
from gf_metric import compute_gamma_factor_for_lists, get_spikes

logger = Logger(log_fname='brian2_network_nevergrad_optimization')

start_scope()
tau = 1 * ms

N = 12; tau_vr = 4.0

GLIF_eqs = '''
dv/dt = ((G * (E_L - v) + R_I * (I_ext + I_syn_tot)) / C_m)/tau : 1
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

neurons = NeuronGroup(N=N, model=GLIF_eqs, threshold='v>30', reset=reset, method='exact')

synapses = Synapses(neurons, neurons, model=synapse_eqs, on_pre='I_syn = I_syn - f_I * I_syn + I_A', method='exact')
synapses.connect(condition=True)

poisson_input_grp = PoissonGroup(N, 30.*Hz)
feedforward = Synapses(poisson_input_grp, neurons, model=in_eqs, on_pre='I_in = 12', method='exact')
feedforward.connect(j='i')

spikemon = SpikeMonitor(neurons[:], 'v', record=True)

store()


def run_simulation_multiobjective(rate, w, C_m, G, R_I, f_v, f_I, E_L, b_s, b_v, a_v, delta_theta_s, delta_V, theta_inf, I_A, t_interval=time_interval*ms):
    restore()
    # start_scope()
    # tau = 1*ms

    # neurons = NeuronGroup(N=N, model=GLIF_eqs, threshold='v>30*mV', reset=reset, method='euler')
    neurons.set_states({'f_I': f_I, 'C_m': C_m, 'G': G, 'R_I': R_I, 'f_v': f_v, 'E_L': E_L, 'b_s': b_s, 'b_v': b_v,
                        'a_v': a_v, 'delta_theta_s': delta_theta_s, 'delta_V': delta_V, 'theta_inf': theta_inf, 'I_A': I_A})

    # PoissonInput(neurons, 'I_ext', N=N, rate=rate, weight=1*mA)
    # poisson_input_grp = PoissonGroup(N, rate/tau)
    # feedforward = Synapses(poisson_input_grp, neurons, model=in_eqs, on_pre='I_in = 1')
    # feedforward.connect(j='i')
    poisson_input_grp.rates = rate * Hz

    # synapses = Synapses(neurons, neurons, model=synapse_eqs, on_pre='I_syn = I_syn - f_I * I_syn + I_A', method='euler')
    # synapses.connect()
    synapses.set_states({'w': w})

    # spikemon = SpikeMonitor(neurons[:], 'v', record=True)

    run(t_interval)
    silent_model = spikemon.num_spikes == 0
    print('DEBUG: spikemon.num_spikes: {}'.format(spikemon.num_spikes))
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
    restore()
    # start_scope()

    # neurons = NeuronGroup(N=N, model=GLIF_eqs, threshold='v>30*mV', reset=reset, method='euler')
    neurons.set_states(
        {'f_I': f_I, 'C_m': C_m, 'G': G, 'R_I': R_I, 'f_v': f_v, 'E_L': E_L, 'b_s': b_s, 'b_v': b_v, 'a_v': a_v,
         'delta_theta_s': delta_theta_s, 'delta_V': delta_V, 'theta_inf': theta_inf, 'I_A': I_A})

    print('E_L', neurons.get_states({'E_L'}))
    # poisson_input_grp = PoissonGroup(N, rate/tau)
    # feedforward = Synapses(poisson_input_grp, neurons, model=in_eqs, on_pre='I_in = 1')
    # feedforward.connect(j='i')
    poisson_input_grp.rates = rate * Hz

    # synapses = Synapses(neurons, neurons, model=synapse_eqs, on_pre='I_syn = I_syn - f_I * I_syn + I_A', method='euler')
    # synapses.connect()
    synapses.set_states({'w': w})

    # spikemon = SpikeMonitor(neurons[:], 'v', record=True)

    run(t_interval)
    print('DEBUG: spikemon.num_spikes: {}'.format(spikemon.num_spikes))
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
    elif loss_fn in ['van_rossum_dist', 'poisson_nll', 'kl_div', 'vrdfrd']:
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


def get_spike_train_for(rate, weights, neurons_params):
    # start_scope()
    # tau = 1*ms
    restore()

    # neurons = NeuronGroup(N=N, model=GLIF_eqs, threshold='v>30', reset=reset, method='euler')
    # neurons_params['R_I'] = neurons_params['R_I']
    # neurons_params['delta_theta_s'] = neurons_params['delta_theta_s']
    # neurons_params['delta_V'] = neurons_params['delta_V']
    # neurons_params['theta_inf'] = neurons_params['theta_inf']
    # neurons_params['E_L'] = neurons_params['E_L']
    # neurons_params['I_A'] = neurons_params['I_A']
    neurons.set_states(neurons_params)

    # poisson_input_grp = PoissonGroup(N, rate*Hz)
    # feedforward = Synapses(poisson_input_grp, neurons, model=in_eqs, on_pre='I_in = 1', method='euler')
    # feedforward.connect(j='i')
    poisson_input_grp.rates = rate * Hz

    # synapses = Synapses(neurons, neurons, model=synapse_eqs, on_pre='I_syn = I_syn - f_I * I_syn + I_A', method='euler')
    # synapses.connect()
    synapses.set_states({'w': weights})

    # spikemon = SpikeMonitor(neurons[:], 'v', record=True)
    mon_in = SpikeMonitor(poisson_input_grp[:], record=True)

    run(time_interval*ms)
    print('spikes:', spikemon.num_spikes)
    print('mon_in spikes:', mon_in.num_spikes)
    # print(synapses.get_states())
    # print(poisson_input_grp.get_states())
    # print(neurons.get_states())

    return torch.tensor(data_util.convert_brian_spike_train_dict_to_boolean_matrix(spikemon.spike_trains(), t_max=time_interval), dtype=torch.float32)
