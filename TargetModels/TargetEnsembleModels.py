import numpy as np
import torch
from torch import tensor as T

from Models.GLIF import GLIF
from Models.LIF import LIF
from experiments import randomise_parameters, zip_tensor_dicts


def glif_ensembles_model_dales_compliant(random_seed, N = 12):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    pop_size = int(N / 3)
    params_pop1 = {'tau_m': 2.5, 'G': 0.9, 'E_L': -40., 'delta_theta_s': 14., 'b_s': 0.4, 'f_v': 0.14, 'delta_V': 10.,
                   'f_I': 0.3, 'I_A': 1.2, 'b_v': 0.4, 'a_v': 0.5, 'theta_inf': -18., 'R_I': 58.}
    weights_std = 0.025
    hand_coded_params_pop1 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                # torch.cat([T(pop_size*[0.06]), T(pop_size*[0.03]), T(pop_size*[0.01])])}
                                                torch.cat([T(pop_size * [0.3]), T(pop_size * [0.2]), T(pop_size * [0.06])])}
                                                # torch.cat([T(pop_size * [0.0]), T(pop_size * [0.0]), T(pop_size * [0.0])])}

    params_pop2 = {'tau_m': 1.4, 'G': 0.8, 'E_L': -52., 'delta_theta_s': 14., 'b_s': 0.4, 'f_v': 0.14, 'delta_V': 12.,
                   'f_I': 0.3, 'I_A': 1., 'b_v': 0.5, 'a_v': 0.4, 'theta_inf': -16., 'R_I': 57.}
    hand_coded_params_pop2 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                # torch.cat([T(4*[.02]), T(4*[.3]), T(4*[0.15])])}
                                                torch.cat([T(4 * [.03]), T(4 * [.3]), T(4 * [0.4])])}
                                                # torch.cat([T(pop_size * [0.0]), T(pop_size * [0.0]), T(pop_size * [0.0])])}

    params_pop3 = {'tau_m': 1.1, 'G': 0.7, 'E_L': -56., 'delta_theta_s': 14., 'b_s': 0.4, 'f_v': 0.14, 'delta_V': 14.,
                   'f_I': 0.5, 'I_A': 1.2, 'b_v': 0.5, 'a_v': 0.4, 'theta_inf': -12., 'R_I': 57.}
    hand_coded_params_pop3 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                # torch.cat([T(4 * [-.4]), T(4 * [-.2]), T(4 * [-0.02])])}
                                                torch.cat([T(4*[-.4]), T(4*[-.2]), T(4*[-0.06])])}
                                                # torch.cat([T(pop_size * [0.0]), T(pop_size * [0.0]), T(pop_size * [0.0])])}

    params_pop1 = randomise_parameters(params_pop1, coeff=T(0.025), N_dim=pop_size)
    params_pop1 = zip_tensor_dicts(params_pop1, hand_coded_params_pop1)
    params_pop2 = randomise_parameters(params_pop2, coeff=T(0.025), N_dim=pop_size)
    params_pop2 = zip_tensor_dicts(params_pop2, hand_coded_params_pop2)
    params_pop3 = randomise_parameters(params_pop3, coeff=T(0.025), N_dim=pop_size)
    params_pop3 = zip_tensor_dicts(params_pop3, hand_coded_params_pop3)
    randomised_params = zip_tensor_dicts(zip_tensor_dicts(params_pop1, params_pop2), params_pop3)

    return GLIF(parameters=randomised_params, N=N,
                neuron_types=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1]))


def lif_ensembles_model_dales_compliant(random_seed, N = 12):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    pop_size = int(N / 3)
    params_pop1 = {'tau_m': 2.3, 'E_L': -38., 'R_I': 150., 'tau_g': 3.5}
    weights_std = 0.025
    hand_coded_params_pop1 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                torch.cat([T(pop_size*[0.3]), T(pop_size*[0.15]), T(pop_size*[0.1])])}
                                                # torch.cat([T(pop_size*[0.0]), T(pop_size*[0.0]), T(pop_size*[0.0])])}

    params_pop2 = {'tau_m': 1.4, 'E_L': -52., 'R_I': 127., 'tau_g': 2.5}
    hand_coded_params_pop2 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                torch.cat([T(4*[.08]), T(4*[.22]), T(4*[0.45])])}
                                                # torch.cat([T(4*[.0]), T(4*[.0]), T(4*[0.0])])}

    params_pop3 = {'tau_m': 1.15, 'E_L': -66., 'R_I': 115., 'tau_g': 2.1}
    hand_coded_params_pop3 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                torch.cat([T(4*[-.4]), T(4*[-.2]), T(4*[-0.05])])}
                                                # torch.cat([T(4*[-.0]), T(4*[-.0]), T(4*[-0.0])])}

    params_pop1 = randomise_parameters(params_pop1, coeff=T(0.025), N_dim=pop_size)
    params_pop1 = zip_tensor_dicts(params_pop1, hand_coded_params_pop1)
    params_pop2 = randomise_parameters(params_pop2, coeff=T(0.025), N_dim=pop_size)
    params_pop2 = zip_tensor_dicts(params_pop2, hand_coded_params_pop2)
    params_pop3 = randomise_parameters(params_pop3, coeff=T(0.025), N_dim=pop_size)
    params_pop3 = zip_tensor_dicts(params_pop3, hand_coded_params_pop3)
    randomised_params = zip_tensor_dicts(zip_tensor_dicts(params_pop1, params_pop2), params_pop3)

    return LIF(parameters=randomised_params, N=N,
               neuron_types=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1]))
