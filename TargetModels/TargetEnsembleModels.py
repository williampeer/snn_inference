import numpy as np
import torch
from torch import tensor as T

from Models.GLIF import GLIF
from Models.LIF import LIF
from Models.LIF_ASC import LIF_ASC
from Models.LIF_R import LIF_R
from Models.LIF_R_ASC import LIF_R_ASC
from experiments import randomise_parameters, zip_tensor_dicts


def glif_ensembles_model_dales_compliant(random_seed, N = 12):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    pop_size = int(N / 3)
    params_pop1 = {'tau_m': 2.5, 'G': 0.8, 'E_L': -40., 'delta_theta_s': 18., 'b_s': 0.4, 'f_v': 0.14, 'delta_V': 10.,
                   'f_I': 0.45, 'I_A': 1.2, 'b_v': 0.4, 'a_v': 0.4, 'theta_inf': -10., 'R_I': 80.}
    weights_std = 0.25
    hand_coded_params_pop1 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                # torch.cat([T(pop_size*[0.06]), T(pop_size*[0.03]), T(pop_size*[0.01])])}
                                                # torch.cat([T(pop_size * [0.2]), T(pop_size * [0.15]), T(pop_size * [0.05])])}
                                                # torch.cat([T(pop_size * [0.25]), T(pop_size * [0.15]), T(pop_size * [0.05])])}
                                                torch.cat([T(pop_size * [0.2]), T(pop_size * [0.1]), T(pop_size * [0.03])])}
                                                # torch.cat([T(pop_size * [0.0]), T(pop_size * [0.0]), T(pop_size * [0.0])])}

    params_pop2 = {'tau_m': 1.5, 'G': 0.8, 'E_L': -49., 'delta_theta_s': 14., 'b_s': 0.3, 'f_v': 0.14, 'delta_V': 12.,
                   'f_I': 0.35, 'I_A': 1.2, 'b_v': 0.3, 'a_v': 0.3, 'theta_inf': -12., 'R_I': 74.}
    hand_coded_params_pop2 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                # torch.cat([T(4*[.02]), T(4*[.3]), T(4*[0.15])])}
                                                # torch.cat([T(4 * [.08]), T(4 * [.38]), T(4 * [0.32])])}
                                                torch.cat([T(pop_size * [0.08]), T(pop_size * [0.4]), T(pop_size * [0.32])])}
                                                # torch.cat([T(pop_size * [0.06]), T(pop_size * [0.3]), T(pop_size * [0.2])])}
                                                # torch.cat([T(pop_size * [0.0]), T(pop_size * [0.0]), T(pop_size * [0.0])])}

    params_pop3 = {'tau_m': 1.16, 'G': 0.8, 'E_L': -62., 'delta_theta_s': 18., 'b_s': 0.25, 'f_v': 0.14, 'delta_V': 12.,
                   'f_I': 0.3, 'I_A': 1.5, 'b_v': 0.3, 'a_v': 0.3, 'theta_inf': -8., 'R_I': 70.}
    hand_coded_params_pop3 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                # torch.cat([T(4 * [-.4]), T(4 * [-.2]), T(4 * [-0.02])])}
                                                # torch.cat([T(4*[-.3]), T(4*[-.28]), T(4*[-0.07])])}
                                                # torch.cat([T(pop_size * [-0.32]), T(pop_size * [-0.25]), T(pop_size * [-0.15])])}
                                                torch.cat([T(4 * [-.4]), T(4 * [-.2]), T(4 * [-0.02])])}
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
    params_pop1 = {'tau_m': 2.2, 'E_L': -38., 'R_I': 148., 'tau_g': 3.5}
    weights_std = 0.25
    hand_coded_params_pop1 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                torch.cat([T(pop_size*[0.28]), T(pop_size*[0.2]), T(pop_size*[0.15])])}
                                                # torch.cat([T(pop_size*[0.0]), T(pop_size*[0.0]), T(pop_size*[0.0])])}

    params_pop2 = {'tau_m': 1.4, 'E_L': -50., 'R_I': 125., 'tau_g': 2.5}
    hand_coded_params_pop2 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                torch.cat([T(4*[.1]), T(4*[.43]), T(4*[0.34])])}
                                                # torch.cat([T(4*[.0]), T(4*[.0]), T(4*[0.0])])}

    params_pop3 = {'tau_m': 1.2, 'E_L': -64., 'R_I': 115., 'tau_g': 1.8}
    hand_coded_params_pop3 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                torch.cat([T(4*[-.16]), T(4*[-.4]), T(4*[-0.12])])}
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


def lif_r_ensembles_model_dales_compliant(random_seed, N = 12):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    pop_size = int(N / 3)
    params_pop1 = {'tau_m': 2.5, 'tau_g': 3.0, 'G': 0.9, 'E_L': -39., 'delta_theta_s': 18., 'b_s': 0.4, 'f_v': 0.14, 'delta_V': 10., 'R_I': 112.}
    weights_std = 0.25
    hand_coded_params_pop1 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2 * weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                # torch.cat([T(pop_size*[0.06]), T(pop_size*[0.03]), T(pop_size*[0.01])])}
                                                # torch.cat([T(pop_size * [0.2]), T(pop_size * [0.15]), T(pop_size * [0.05])])}
                                                torch.cat([T(pop_size * [0.2]), T(pop_size * [0.1]), T(pop_size * [0.03])])}
                                                # torch.cat([T(pop_size * [0.0]), T(pop_size * [0.0]), T(pop_size * [0.0])])}

    params_pop2 = {'tau_m': 1.5, 'tau_g': 2.4, 'G': 0.8, 'E_L': -49., 'delta_theta_s': 14., 'b_s': 0.3, 'f_v': 0.14, 'delta_V': 12., 'R_I': 88.}
    hand_coded_params_pop2 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2 * weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                # torch.cat([T(4*[.02]), T(4*[.3]), T(4*[0.15])])}
                                                # torch.cat([T(4 * [.08]), T(4 * [.38]), T(4 * [0.32])])}
                                                torch.cat([T(pop_size * [0.04]), T(pop_size * [0.4]), T(pop_size * [0.2])])}
                                                # torch.cat([T(pop_size * [0.0]), T(pop_size * [0.0]), T(pop_size * [0.0])])}

    params_pop3 = {'tau_m': 1.16, 'tau_g': 1.8, 'G': 0.7, 'E_L': -61., 'delta_theta_s': 18., 'b_s': 0.25, 'f_v': 0.14, 'delta_V': 12., 'R_I': 78.}
    hand_coded_params_pop3 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2 * weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                torch.cat([T(4 * [-.4]), T(4 * [-.2]), T(4 * [-0.02])])}
                                                # torch.cat([T(4*[-.3]), T(4*[-.28]), T(4*[-0.07])])}
                                                # torch.cat([T(pop_size * [-0.32]), T(pop_size * [-0.25]), T(pop_size * [-0.15])])}
                                                # torch.cat([T(pop_size * [0.0]), T(pop_size * [0.0]), T(pop_size * [0.0])])}

    params_pop1 = randomise_parameters(params_pop1, coeff=T(0.025), N_dim=pop_size)
    params_pop1 = zip_tensor_dicts(params_pop1, hand_coded_params_pop1)
    params_pop2 = randomise_parameters(params_pop2, coeff=T(0.025), N_dim=pop_size)
    params_pop2 = zip_tensor_dicts(params_pop2, hand_coded_params_pop2)
    params_pop3 = randomise_parameters(params_pop3, coeff=T(0.025), N_dim=pop_size)
    params_pop3 = zip_tensor_dicts(params_pop3, hand_coded_params_pop3)
    randomised_params = zip_tensor_dicts(zip_tensor_dicts(params_pop1, params_pop2), params_pop3)

    return LIF_R(parameters=randomised_params, N=N,
                 neuron_types=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1]))


def lif_asc_ensembles_model_dales_compliant(random_seed, N = 12):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    pop_size = int(N / 3)
    params_pop1 = {'tau_m': 2.2, 'E_L': -40., 'f_I': 0.45, 'G': 0.9, 'delta_theta_s': 18., 'b_s': 0.4, 'f_v': 0.14,
                   'delta_V': 10., 'I_A': 1.2, 'R_I': 95.}
    weights_std = 0.25
    hand_coded_params_pop1 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                # torch.cat([T(pop_size*[0.28]), T(pop_size*[0.2]), T(pop_size*[0.15])])}
                                                torch.cat([T(pop_size * [0.2]), T(pop_size * [0.1]), T(pop_size * [0.03])])}
                                                # torch.cat([T(pop_size*[0.0]), T(pop_size*[0.0]), T(pop_size*[0.0])])}

    params_pop2 = {'tau_m': 1.4, 'E_L': -50., 'f_I': 0.35, 'G': 0.8, 'delta_theta_s': 14., 'b_s': 0.3, 'f_v': 0.14,
                   'delta_V': 12., 'I_A': 1.2, 'R_I': 86.}
    hand_coded_params_pop2 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                # torch.cat([T(4*[.1]), T(4*[.43]), T(4*[0.34])])}
                                                torch.cat([T(pop_size * [0.06]), T(pop_size * [0.3]), T(pop_size * [0.2])])}
                                                # torch.cat([T(4*[.0]), T(4*[.0]), T(4*[0.0])])}

    params_pop3 = {'tau_m': 1.2, 'E_L': -61., 'f_I': 0.3, 'G': 0.7, 'delta_theta_s': 18., 'b_s': 0.25, 'f_v': 0.14,
                   'delta_V': 12., 'I_A': 1.5, 'R_I': 80.}
    hand_coded_params_pop3 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                # torch.cat([T(4*[-.16]), T(4*[-.4]), T(4*[-0.12])])}
                                                torch.cat([T(4 * [-.4]), T(4 * [-.2]), T(4 * [-0.02])])}
                                                # torch.cat([T(4*[-.0]), T(4*[-.0]), T(4*[-0.0])])}

    params_pop1 = randomise_parameters(params_pop1, coeff=T(0.025), N_dim=pop_size)
    params_pop1 = zip_tensor_dicts(params_pop1, hand_coded_params_pop1)
    params_pop2 = randomise_parameters(params_pop2, coeff=T(0.025), N_dim=pop_size)
    params_pop2 = zip_tensor_dicts(params_pop2, hand_coded_params_pop2)
    params_pop3 = randomise_parameters(params_pop3, coeff=T(0.025), N_dim=pop_size)
    params_pop3 = zip_tensor_dicts(params_pop3, hand_coded_params_pop3)
    randomised_params = zip_tensor_dicts(zip_tensor_dicts(params_pop1, params_pop2), params_pop3)

    return LIF_ASC(parameters=randomised_params, N=N,
                   neuron_types=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1]))


def lif_r_asc_ensembles_model_dales_compliant(random_seed, N = 12):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    pop_size = int(N / 3)
    params_pop1 = {'tau_m': 2.2, 'E_L': -40., 'f_I': 0.45, 'G': 0.9, 'delta_theta_s': 18., 'b_s': 0.4, 'f_v': 0.14,
                   'delta_V': 10., 'I_A': 1.2, 'R_I': 95.}
    weights_std = 0.25
    hand_coded_params_pop1 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2 * weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                # torch.cat([T(pop_size*[0.28]), T(pop_size*[0.2]), T(pop_size*[0.15])])}
                                                torch.cat([T(pop_size * [0.2]), T(pop_size * [0.1]), T(pop_size * [0.03])])}
                                                # torch.cat([T(pop_size*[0.0]), T(pop_size*[0.0]), T(pop_size*[0.0])])}

    params_pop2 = {'tau_m': 1.4, 'E_L': -50., 'f_I': 0.35, 'G': 0.8, 'delta_theta_s': 14., 'b_s': 0.3, 'f_v': 0.14,
                   'delta_V': 12., 'I_A': 1.2, 'R_I': 86.}
    hand_coded_params_pop2 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2 * weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                # torch.cat([T(4*[.1]), T(4*[.43]), T(4*[0.34])])}
                                                torch.cat([T(pop_size * [0.06]), T(pop_size * [0.3]), T(pop_size * [0.2])])}
                                                # torch.cat([T(4*[.0]), T(4*[.0]), T(4*[0.0])])}

    params_pop3 = {'tau_m': 1.2, 'E_L': -61., 'f_I': 0.3, 'G': 0.7, 'delta_theta_s': 18., 'b_s': 0.25, 'f_v': 0.14,
                   'delta_V': 12., 'I_A': 1.5, 'R_I': 80.}
    hand_coded_params_pop3 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2 * weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                # torch.cat([T(4*[-.16]), T(4*[-.4]), T(4*[-0.12])])}
                                                torch.cat([T(4 * [-.4]), T(4 * [-.2]), T(4 * [-0.02])])}
                                                # torch.cat([T(4*[-.0]), T(4*[-.0]), T(4*[-0.0])])}

    params_pop1 = randomise_parameters(params_pop1, coeff=T(0.025), N_dim=pop_size)
    params_pop1 = zip_tensor_dicts(params_pop1, hand_coded_params_pop1)
    params_pop2 = randomise_parameters(params_pop2, coeff=T(0.025), N_dim=pop_size)
    params_pop2 = zip_tensor_dicts(params_pop2, hand_coded_params_pop2)
    params_pop3 = randomise_parameters(params_pop3, coeff=T(0.025), N_dim=pop_size)
    params_pop3 = zip_tensor_dicts(params_pop3, hand_coded_params_pop3)
    randomised_params = zip_tensor_dicts(zip_tensor_dicts(params_pop1, params_pop2), params_pop3)

    return LIF_R_ASC(parameters=randomised_params, N=N,
                     neuron_types=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1]))
