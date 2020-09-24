import torch

from Dev.brian_GLIF_model_setup import *
from Dev.setup_data_for_brian import *
from Log import Logger
from eval import calculate_loss
from gf_metric import compute_gamma_factor_for_lists, get_spikes

logger = Logger(log_fname='brian2_network_nevergrad_optimization')
tau_vr = 4.0


def run_simulation_multiobjective(w, C_m, G, R_I, f_v, f_I, E_L, b_s, b_v, a_v, delta_theta_s, delta_V, theta_inf, I_A, t_interval=time_interval*ms):
    restore('init')

    synapses.set_states({ 'w': w })
    neurons.set_states({ 'f_I': f_I, 'C_m': C_m, 'G': G, 'R_I': R_I*ohm, 'f_v': f_v, 'E_L': E_L*mV,
                         'b_s': b_s, 'b_v': b_v, 'a_v': a_v, 'delta_theta_s': delta_theta_s * mV, 'delta_V': delta_V * mV, 'theta_inf': theta_inf * mV, 'I_A': I_A*mA})

    in_grp.set_spikes(np.reshape(input_indices, (-1,)), np.reshape(input_times*ms, (-1,)))
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
    restore('init')

    synapses.set_states({ 'w': w })
    neurons.set_states({'f_I': f_I, 'C_m': C_m, 'G': G, 'R_I': R_I * ohm, 'f_v': f_v, 'E_L': E_L * mV,
                        'b_s': b_s, 'b_v': b_v, 'a_v': a_v, 'delta_theta_s': delta_theta_s * mV, 'delta_V': delta_V * mV, 'theta_inf': theta_inf * mV, 'I_A': I_A*mA})

    in_grp.set_spikes(np.reshape(input_indices, (-1,)), np.reshape(input_times*ms, (-1,)))  # TODO: "fix"
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
