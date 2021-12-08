import numpy as np
import torch
from torch import tensor as T

from Models.GLIF import GLIF
from Models.LIF import LIF
from Models.no_grad.LIF_R_ASC_no_grad import LIF_R_ASC_no_grad
from Models.no_grad.LIF_R_no_grad import LIF_R_no_grad
from experiments import randomise_parameters, zip_tensor_dicts


# def lif_continuous_ensembles_model_dales_compliant(random_seed, N = 12):
#     torch.manual_seed(random_seed)
#     np.random.seed(random_seed)
#
#     weights_std = 0.05
#     pop_size = int(N / 3)
#     pop_size_last = N - 2*pop_size
#
#     params_pop1 = {'tau_m': 4.6, 'E_L': -50., 'tau_s': 6.5, 'spike_threshold': 30.}
#     hand_coded_params_pop1 = {'preset_weights': (torch.ones((pop_size, 1)) +
#                                                  (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
#                                                 torch.cat([T(pop_size * [0.15]), T(pop_size * [0.1]), T(pop_size_last * [0.06])])}
#
#     params_pop2 = {'tau_m': 3.5, 'E_L': -60., 'tau_s': 4.5, 'spike_threshold': 30.}
#     hand_coded_params_pop2 = {'preset_weights': (torch.ones((pop_size, 1)) + (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
#                                                 torch.cat([T(pop_size * [.08]), T(pop_size * [.3]), T(pop_size_last * [.15])])}
#
#     params_pop3 = {'tau_m': 3.5, 'E_L': -68., 'tau_s': 2.5, 'spike_threshold': 30.}
#     hand_coded_params_pop3 = {'preset_weights': (torch.ones((pop_size_last, 1)) +
#                                                  (2*weights_std * torch.randn((pop_size_last, N))) - weights_std) *
#                                                 torch.cat([T(pop_size * [-.06]), T(pop_size * [-.1]), T(pop_size_last * [-.1])])}
#
#     params_pop1 = randomise_parameters(params_pop1, coeff=T(0.025), N_dim=pop_size)
#     params_pop1 = zip_tensor_dicts(params_pop1, hand_coded_params_pop1)
#     params_pop2 = randomise_parameters(params_pop2, coeff=T(0.025), N_dim=pop_size)
#     params_pop2 = zip_tensor_dicts(params_pop2, hand_coded_params_pop2)
#     params_pop3 = randomise_parameters(params_pop3, coeff=T(0.025), N_dim=pop_size_last)
#     params_pop3 = zip_tensor_dicts(params_pop3, hand_coded_params_pop3)
#     randomised_params = zip_tensor_dicts(zip_tensor_dicts(params_pop1, params_pop2), params_pop3)
#
#     neuron_types = np.ones((N,))
#     for i in range(int(N / 3)):
#         neuron_types[-(1 + i)] = -1
#     return LIF(parameters=randomised_params, N=N, neuron_types=neuron_types)
#
#
# def lif_r_continuous_ensembles_model_dales_compliant(random_seed, N = 12):
#     torch.manual_seed(random_seed)
#     np.random.seed(random_seed)
#
#     pop_size = int(N / 3)
#     pop_size_last = N - 2*pop_size
#     # params_pop1 = {'tau_m': 2.5, 'tau_s': 3.0, 'G': 0.9, 'E_L': -39., 'delta_theta_s': 18., 'b_s': 0.4, 'f_v': 0.14, 'delta_V': 10.}
#     params_pop1 = {'tau_m': 5.2, 'E_L': -48., 'G': 0.85, 'delta_theta_s': 18., 'b_s': 0.4, 'f_v': 0.14, 'delta_V': 10., 'tau_s': 7.}
#     weights_std = 0.05
#     # weights_std = 0
#     hand_coded_params_pop1 = {'preset_weights': (torch.ones((pop_size, 1)) +
#                                                  (2 * weights_std * torch.randn((pop_size, N))) - weights_std) *
#                                                 # torch.cat([T(pop_size*[0.06]), T(pop_size*[0.03]), T(pop_size*[0.01])])}
#                                                 # torch.cat([T(pop_size * [0.2]), T(pop_size * [0.15]), T(pop_size * [0.05])])}
#                                                 # torch.cat([T(pop_size * [0.2]), T(pop_size * [0.1]), T(pop_size * [0.03])])}
#                                                 # torch.cat([T(pop_size * [0.0]), T(pop_size * [0.0]), T(pop_size * [0.0])])}
#                                                 torch.cat([T(pop_size * [0.12]), T(pop_size * [0.08]), T(pop_size_last * [0.06])])}
#
#     # params_pop2 = {'tau_m': 1.5, 'tau_s': 2.4, 'G': 0.8, 'E_L': -49., 'delta_theta_s': 14., 'b_s': 0.3, 'f_v': 0.14, 'delta_V': 12.}
#     params_pop2 = {'tau_m': 4.3, 'E_L': -55., 'G': 0.8, 'delta_theta_s': 14., 'b_s': 0.3, 'f_v': 0.14, 'delta_V': 12., 'tau_s': 4.5}
#     hand_coded_params_pop2 = {'preset_weights': (torch.ones((pop_size, 1)) +
#                                                  (2 * weights_std * torch.randn((pop_size, N))) - weights_std) *
#                                                 # torch.cat([T(4*[.02]), T(4*[.3]), T(4*[0.15])])}
#                                                 # torch.cat([T(pop_size * [.08]), T(pop_size * [.38]), T(pop_size * [0.32])])}
#                                                 # torch.cat([T(pop_size * [0.04]), T(pop_size * [0.4]), T(pop_size * [0.2])])}
#                                                 # torch.cat([T(pop_size * [0.0]), T(pop_size * [0.0]), T(pop_size * [0.0])])}
#                                                 torch.cat([T(pop_size * [.1]), T(pop_size * [.4]), T(pop_size_last * [.2])])}
#
#     # params_pop3 = {'tau_m': 1.16, 'tau_s': 1.8, 'G': 0.7, 'E_L': -61., 'delta_theta_s': 18., 'b_s': 0.25, 'f_v': 0.14, 'delta_V': 12.}
#     params_pop3 = {'tau_m': 4.1, 'E_L': -68., 'G': 0.8, 'delta_theta_s': 18., 'b_s': 0.25, 'f_v': 0.14, 'delta_V': 12., 'tau_s': 3.3}
#     hand_coded_params_pop3 = {'preset_weights': (torch.ones((pop_size_last, 1)) +
#                                                  (2 * weights_std * torch.randn((pop_size_last, N))) - weights_std) *
#                                                 # torch.cat([T(pop_size * [-.4]), T(pop_size * [-.2]), T(pop_size * [-0.02])])}
#                                                 # torch.cat([T(4*[-.3]), T(4*[-.28]), T(4*[-0.07])])}
#                                                 # torch.cat([T(pop_size * [-0.32]), T(pop_size * [-0.25]), T(pop_size * [-0.15])])}
#                                                 # torch.cat([T(pop_size * [0.0]), T(pop_size * [0.0]), T(pop_size * [0.0])])}
#                                                 torch.cat([T(pop_size * [-.15]), T(pop_size * [-.2]), T(pop_size_last * [-.3])])}
#
#     params_pop1 = randomise_parameters(params_pop1, coeff=T(0.025), N_dim=pop_size)
#     params_pop1 = zip_tensor_dicts(params_pop1, hand_coded_params_pop1)
#     params_pop2 = randomise_parameters(params_pop2, coeff=T(0.025), N_dim=pop_size)
#     params_pop2 = zip_tensor_dicts(params_pop2, hand_coded_params_pop2)
#     params_pop3 = randomise_parameters(params_pop3, coeff=T(0.025), N_dim=pop_size_last)
#     params_pop3 = zip_tensor_dicts(params_pop3, hand_coded_params_pop3)
#     randomised_params = zip_tensor_dicts(zip_tensor_dicts(params_pop1, params_pop2), params_pop3)
#
#     neuron_types = np.ones((N,))
#     for i in range(int(N / 3)):
#         neuron_types[-(1 + i)] = -1
#     return LIF_R_no_grad(parameters=randomised_params, N=N, neuron_types=neuron_types)
#
#
# def lif_r_asc_continuous_ensembles_model_dales_compliant(random_seed, N = 12):
#     torch.manual_seed(random_seed)
#     np.random.seed(random_seed)
#
#     pop_size = int(N / 3)
#     pop_size_last = N - 2*pop_size
#     params_pop1 = {'tau_m': 5.5, 'E_L': -54., 'f_I': 0.45, 'G': 0.9, 'delta_theta_s': 18., 'b_s': 0.4, 'f_v': 0.14,
#                    'delta_V': 10., 'tau_s': 6.5}
#     weights_std = 0.05
#     # weights_std = 0
#     hand_coded_params_pop1 = {'preset_weights': (torch.ones((pop_size, 1)) +
#                                                  (2 * weights_std * torch.randn((pop_size, N))) - weights_std) *
#                                                 # torch.cat([T(pop_size*[0.28]), T(pop_size*[0.2]), T(pop_size*[0.15])])}
#                                                 # torch.cat([T(pop_size * [0.2]), T(pop_size * [0.1]), T(pop_size * [0.03])])}
#                                                 # torch.cat([T(pop_size*[0.0]), T(pop_size*[0.0]), T(pop_size*[0.0])])}
#                                                 torch.cat([T(pop_size * [0.1]), T(pop_size * [0.08]), T(pop_size_last * [0.06])])}
#
#     params_pop2 = {'tau_m': 4.5, 'E_L': -58., 'f_I': 0.35, 'G': 0.85, 'delta_theta_s': 16., 'b_s': 0.3, 'f_v': 0.14,
#                    'delta_V': 12., 'tau_s': 4.5}
#     hand_coded_params_pop2 = {'preset_weights': (torch.ones((pop_size, 1)) +
#                                                  (2 * weights_std * torch.randn((pop_size, N))) - weights_std) *
#                                                 # torch.cat([T(4*[.1]), T(4*[.43]), T(4*[0.34])])}
#                                                 # torch.cat([T(pop_size * [0.06]), T(pop_size * [0.3]), T(pop_size * [0.2])])}
#                                                 # torch.cat([T(4*[.0]), T(4*[.0]), T(4*[0.0])])}
#                                                 torch.cat([T(pop_size * [.08]), T(pop_size * [.3]), T(pop_size_last * [.2])])}
#
#     params_pop3 = {'tau_m': 4.1, 'E_L': -68., 'f_I': 0.35, 'G': 0.9, 'delta_theta_s': 18., 'b_s': 0.25, 'f_v': 0.14,
#                    'delta_V': 12., 'tau_s': 3.2}
#     hand_coded_params_pop3 = {'preset_weights': (torch.ones((pop_size_last, 1)) +
#                                                  (2 * weights_std * torch.randn((pop_size_last, N))) - weights_std) *
#                                                 # torch.cat([T(4*[-.16]), T(4*[-.4]), T(4*[-0.12])])}
#                                                 # torch.cat([T(pop_size * [-.4]), T(pop_size * [-.2]), T(pop_size * [-0.02])])}
#                                                 # torch.cat([T(4*[-.0]), T(4*[-.0]), T(4*[-0.0])])}
#                                                 torch.cat([T(pop_size * [-.04]), T(pop_size * [-.08]), T(pop_size_last * [-.12])])}
#
#     params_pop1 = randomise_parameters(params_pop1, coeff=T(0.025), N_dim=pop_size)
#     params_pop1 = zip_tensor_dicts(params_pop1, hand_coded_params_pop1)
#     params_pop2 = randomise_parameters(params_pop2, coeff=T(0.025), N_dim=pop_size)
#     params_pop2 = zip_tensor_dicts(params_pop2, hand_coded_params_pop2)
#     params_pop3 = randomise_parameters(params_pop3, coeff=T(0.025), N_dim=pop_size_last)
#     params_pop3 = zip_tensor_dicts(params_pop3, hand_coded_params_pop3)
#     randomised_params = zip_tensor_dicts(zip_tensor_dicts(params_pop1, params_pop2), params_pop3)
#
#     neuron_types = np.ones((N,))
#     for i in range(int(N / 3)):
#         neuron_types[-(1 + i)] = -1
#     return LIF_R_ASC_no_grad(parameters=randomised_params, N=N, neuron_types=neuron_types)


def glif(random_seed, N = 12):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    pop_size = int(N / 2)
    pop_size_last = N - pop_size
    params_pop1 = {'tau_m': 3.4, 'G': 0.7, 'E_L': -52., 'delta_theta_s': 18., 'b_s': 0.4, 'f_v': 0.14, 'delta_V': 10.,
                   'f_I': 0.5, 'b_v': 0.3, 'a_v': 0.2, 'theta_inf': -12., 'tau_s': 4.5}
    weights_std = 0.05
    # weights_std = 0
    hand_coded_params_pop1 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                torch.cat([T(pop_size * [0.55]), T(pop_size_last * [0.45])])}

    params_pop2 = {'tau_m': 2.6, 'G': 0.8, 'E_L': -62., 'delta_theta_s': 18., 'b_s': 0.25, 'f_v': 0.14, 'delta_V': 12.,
                   'f_I': 0.35, 'b_v': 0.4, 'a_v': 0.3, 'theta_inf': -11., 'tau_s': 2.4}
    hand_coded_params_pop2 = {'preset_weights': (torch.ones((pop_size_last, 1)) +
                                                 (2*weights_std * torch.randn((pop_size_last, N))) - weights_std) *
                                                torch.cat([T(pop_size * [-.5]), T(pop_size_last * [-.35])])}

    params_pop1 = randomise_parameters(params_pop1, coeff=T(0.025), N_dim=pop_size)
    params_pop1 = zip_tensor_dicts(params_pop1, hand_coded_params_pop1)
    params_pop2 = randomise_parameters(params_pop2, coeff=T(0.025), N_dim=pop_size)
    params_pop2 = zip_tensor_dicts(params_pop2, hand_coded_params_pop2)
    randomised_params = zip_tensor_dicts(params_pop1, params_pop2)

    neuron_types = np.ones((N,))
    for i in range(int(N / 2)):
        neuron_types[-(1 + i)] = -1
    return GLIF(parameters=randomised_params, N=N, neuron_types=neuron_types)
