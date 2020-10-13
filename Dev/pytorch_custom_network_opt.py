from brian2 import *

from Log import Logger
from Models.GLIF import GLIF
from eval import calculate_loss
from experiments import generate_synthetic_data

logger = Logger(log_fname='pytorch_nevergrad_optimization')
tau_vr = 4.0


def in_place_cast_to_float32(np_dict):
    for key in np_dict.keys():
        np_dict[key] = np.array(np_dict[key], dtype='float32')


def pytorch_run_simulation_for(rate, w, C_m, G, R_I, f_v, f_I, E_L, b_s, b_v, a_v, delta_theta_s, delta_V, theta_inf,
                               I_A, loss_fn,target_model, target_rate, time_interval=4000):

    model = GLIF({'f_I': np.array(f_I, dtype='float32'), 'C_m': np.array(C_m, dtype='float32'), 'G': np.array(G, dtype='float32'),
                  'R_I': np.array(R_I, dtype='float32'), 'f_v': np.array(f_v, dtype='float32'), 'E_L': np.array(E_L, dtype='float32'),
                  'b_s': np.array(b_s, dtype='float32'), 'b_v': np.array(b_v, dtype='float32'), 'a_v': np.array(a_v, dtype='float32'),
                  'delta_theta_s': np.array(delta_theta_s, dtype='float32'), 'delta_V': np.array(delta_V, dtype='float32'),
                  'theta_inf': np.array(theta_inf, dtype='float32'), 'I_A': np.array(I_A, dtype='float32'),
                  'preset_weights': np.array(w, dtype='float32')})

    model_outputs = generate_synthetic_data(model, rate, time_interval)
    target_outputs = generate_synthetic_data(target_model, target_rate, time_interval)

    loss = np.float(calculate_loss(model_outputs, target_outputs, loss_fn=loss_fn, tau_vr=tau_vr))
    logger.log('loss_fn: {}, loss: {:3.3f}'.format(loss_fn, loss))
    return loss


def get_spike_train_for(rate, neurons_params, run_time=4000):
    in_place_cast_to_float32(neurons_params)
    model = GLIF(neurons_params)
    return generate_synthetic_data(model, rate, run_time)


# def get_spike_train_for_matlab_export(rate, weights, neurons_params, run_time=60*1000):
#     restore()
#
#     neurons.set_states(neurons_params)
#
#     poisson_input_grp.rates = rate * Hz
#
#     synapses.set_states({'w': weights})
#
#     run(run_time * ms)
#     print('spikes:', spikemon.num_spikes)
#
#     return spikemon.spike_trains()
