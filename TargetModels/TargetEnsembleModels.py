import numpy as np
import torch
from torch import tensor as T

from Models.GLIF import GLIF
from Models.LIF import LIF
from experiments import randomise_parameters, zip_tensor_dicts


def glif_ensembles_model(random_seed, N = 12):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    pop_size = int(N / 3)
    params_pop1 = {'C_m': 2.3, 'G': 0.9, 'E_L': -38., 'delta_theta_s': 14., 'b_s': 0.4, 'f_v': 0.14, 'delta_V': 12.,
                   'f_I': 0.6, 'I_A': 2., 'b_v': 0.5, 'a_v': 0.5, 'theta_inf': -24., 'R_I': 130.}
    weights_std = 0.025
    hand_coded_params_pop1 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                torch.cat([T(pop_size*[0.06]), T(pop_size*[0.03]), T(pop_size*[0.01])])}

    params_pop2 = {'C_m': 1.3, 'G': 0.8, 'E_L': -44., 'delta_theta_s': 14., 'b_s': 0.4, 'f_v': 0.14, 'delta_V': 10.,
                   'f_I': 0.4, 'I_A': 1.4, 'b_v': 0.5, 'a_v': 0.5, 'theta_inf': -22., 'R_I': 130.}
    hand_coded_params_pop2 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                torch.cat([T(4*[-.04]), T(4*[.3]), T(4*[0.15])])}

    params_pop3 = {'C_m': 1.1, 'G': 0.7, 'E_L': -54., 'delta_theta_s': 18., 'b_s': 0.4, 'f_v': 0.14, 'delta_V': 10.,
                   'f_I': 0.2, 'I_A': 1.2, 'b_v': 0.5, 'a_v': 0.5, 'theta_inf': -23., 'R_I': 130.}
    hand_coded_params_pop3 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                torch.cat([T(4*[-.2]), T(4*[-.1]), T(4*[0.2])])}

    params_pop1 = randomise_parameters(params_pop1, coeff=T(0.025), N_dim=pop_size)
    params_pop1 = zip_tensor_dicts(params_pop1, hand_coded_params_pop1)
    params_pop2 = randomise_parameters(params_pop2, coeff=T(0.025), N_dim=pop_size)
    params_pop2 = zip_tensor_dicts(params_pop2, hand_coded_params_pop2)
    params_pop3 = randomise_parameters(params_pop3, coeff=T(0.025), N_dim=pop_size)
    params_pop3 = zip_tensor_dicts(params_pop3, hand_coded_params_pop3)
    randomised_params = zip_tensor_dicts(zip_tensor_dicts(params_pop1, params_pop2), params_pop3)

    return GLIF(parameters=randomised_params, N=N)


def glif_ensembles_model_dales_compliant(random_seed, N = 12):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    pop_size = int(N / 3)
    params_pop1 = {'C_m': 2.3, 'G': 0.9, 'E_L': -38., 'delta_theta_s': 14., 'b_s': 0.4, 'f_v': 0.14, 'delta_V': 12.,
                   'f_I': 0.6, 'I_A': 2., 'b_v': 0.5, 'a_v': 0.5, 'theta_inf': -24., 'R_I': 130.}
    weights_std = 0.025
    hand_coded_params_pop1 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                torch.cat([T(pop_size*[0.06]), T(pop_size*[0.03]), T(pop_size*[0.01])])}

    params_pop2 = {'C_m': 1.3, 'G': 0.8, 'E_L': -44., 'delta_theta_s': 14., 'b_s': 0.4, 'f_v': 0.14, 'delta_V': 10.,
                   'f_I': 0.4, 'I_A': 1.4, 'b_v': 0.5, 'a_v': 0.5, 'theta_inf': -22., 'R_I': 130.}
    hand_coded_params_pop2 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                torch.cat([T(4*[.02]), T(4*[.3]), T(4*[0.15])])}

    params_pop3 = {'C_m': 1.1, 'G': 0.7, 'E_L': -54., 'delta_theta_s': 18., 'b_s': 0.4, 'f_v': 0.14, 'delta_V': 10.,
                   'f_I': 0.2, 'I_A': 1.2, 'b_v': 0.5, 'a_v': 0.5, 'theta_inf': -23., 'R_I': 130.}
    hand_coded_params_pop3 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                torch.cat([T(4*[-.2]), T(4*[-.1]), T(4*[-0.02])])}

    params_pop1 = randomise_parameters(params_pop1, coeff=T(0.025), N_dim=pop_size)
    params_pop1 = zip_tensor_dicts(params_pop1, hand_coded_params_pop1)
    params_pop2 = randomise_parameters(params_pop2, coeff=T(0.025), N_dim=pop_size)
    params_pop2 = zip_tensor_dicts(params_pop2, hand_coded_params_pop2)
    params_pop3 = randomise_parameters(params_pop3, coeff=T(0.025), N_dim=pop_size)
    params_pop3 = zip_tensor_dicts(params_pop3, hand_coded_params_pop3)
    randomised_params = zip_tensor_dicts(zip_tensor_dicts(params_pop1, params_pop2), params_pop3)

    return GLIF(parameters=randomised_params, N=N,
                neuron_types=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1]))


def lif_ensembles_model_dales_compliant(random_seed, N = 12):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    pop_size = int(N / 3)
    params_pop1 = {'C_m': 2.3, 'E_L': -38., 'R_I': 100., 'tau_g': 3.}
    weights_std = 0.025
    hand_coded_params_pop1 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                torch.cat([T(pop_size*[0.06]), T(pop_size*[0.03]), T(pop_size*[0.01])])}

    params_pop2 = {'C_m': 1.3, 'E_L': -44., 'R_I': 89., 'tau_g': 3.}
    hand_coded_params_pop2 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                torch.cat([T(4*[.02]), T(4*[.3]), T(4*[0.15])])}

    params_pop3 = {'C_m': 1.1, 'E_L': -60., 'R_I': 80., 'tau_g': 3.}
    hand_coded_params_pop3 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                torch.cat([T(4*[-.2]), T(4*[-.1]), T(4*[-0.02])])}

    params_pop1 = randomise_parameters(params_pop1, coeff=T(0.025), N_dim=pop_size)
    params_pop1 = zip_tensor_dicts(params_pop1, hand_coded_params_pop1)
    params_pop2 = randomise_parameters(params_pop2, coeff=T(0.025), N_dim=pop_size)
    params_pop2 = zip_tensor_dicts(params_pop2, hand_coded_params_pop2)
    params_pop3 = randomise_parameters(params_pop3, coeff=T(0.025), N_dim=pop_size)
    params_pop3 = zip_tensor_dicts(params_pop3, hand_coded_params_pop3)
    randomised_params = zip_tensor_dicts(zip_tensor_dicts(params_pop1, params_pop2), params_pop3)

    return LIF(parameters=randomised_params, N=N,
               neuron_types=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1]))
