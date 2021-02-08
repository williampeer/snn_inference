import numpy as np
import torch
from torch import tensor as T

from Models.LIF import LIF
from experiments import randomise_parameters, zip_tensor_dicts


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
