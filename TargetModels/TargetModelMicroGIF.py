import numpy as np
import torch
from torch import tensor as T

from Models.microGIF import microGIF
from experiments import randomise_parameters, zip_tensor_dicts


def glif_soft_continuous_population_models(random_seed, pop_size=1, N_pops=2):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    weights_std = 0.0
    # weights_std = 0

    N = N_pops * pop_size

    if N_pops == 2:
        weights_excit_1 = (torch.ones((pop_size, 1)) + (2 * weights_std * torch.randn((pop_size, N))) - weights_std) * \
                          torch.cat([T(pop_size * [2.482]), T(pop_size * [1.245])])
        weights_inhib_1 = (torch.ones((pop_size, 1)) + (2 * weights_std * torch.randn((pop_size, N))) - weights_std) * \
                          torch.cat([T(pop_size * [-4.964]), T(pop_size * [-4.964])])

        # Excitatory:
        params_pop_excit_1 = {'tau_m': 10., 'tau_s': 3., 'Delta_noise': 5., 'tau_theta': 1000., 'c': 0.1,
                              'J_theta': 1., 'E_L': 20., 'R_m': 19. }
        hand_coded_params_pop_excit_1 = {'preset_weights': weights_excit_1}

        # Inhibitory
        params_pop_inhib_1 = {'tau_m': 10., 'tau_s': 6., 'Delta_noise': 5., 'tau_theta': 1000., 'c': 0.1,
                              'J_theta': 1., 'E_L': 19.5, 'R_m': 11.964 }
        hand_coded_params_pop_inhib_1 = {'preset_weights': weights_inhib_1}

        params_pop_excit_1 = randomise_parameters(params_pop_excit_1, coeff=T(0.025), N_dim=pop_size)
        params_pop_excit_1 = zip_tensor_dicts(params_pop_excit_1, hand_coded_params_pop_excit_1)
        params_pop_inhib_1 = randomise_parameters(params_pop_inhib_1, coeff=T(0.025), N_dim=pop_size)
        params_pop_inhib_1 = zip_tensor_dicts(params_pop_inhib_1, hand_coded_params_pop_inhib_1)

        randomised_params = zip_tensor_dicts(params_pop_excit_1, params_pop_inhib_1)
    elif N_pops == 4:
        # up to 4 populations, TODO: test values
        weights_excit_1 = (torch.ones((pop_size, 1)) + (
                    2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
            [T(pop_size * [1.245]), T(pop_size * [1.245]), T(pop_size * [1.245]), T(pop_size * [1.245])])
        weights_excit_2 = (torch.ones((pop_size, 1)) + (
                    2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
            [T(pop_size * [1.245]), T(pop_size * [1.245]), T(pop_size * [2.482]), T(pop_size * [1.245])])
        weights_inhib_1 = (torch.ones((pop_size, 1)) + (
                    2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
            [T(pop_size * [-4.964]), T(pop_size * [-4.964]), T(pop_size * [-4.964]), T(pop_size * [-4.964])])
        weights_inhib_2 = (torch.ones((pop_size, 1)) + (
                    2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
            [T(pop_size * [-4.964]), T(pop_size * [-4.964]), T(pop_size * [-4.964]), T(pop_size * [-4.964])])

        # Excitatory:
        params_pop_excit_1 = {'tau_m': 10., 'tau_s': 3., 'Delta_noise': 5., 'tau_theta': 1000., 'c': 10.}
        hand_coded_params_pop_excit_1 = {'preset_weights': weights_excit_1 }
        params_pop_excit_2 = {'tau_m': 10., 'tau_s': 3., 'Delta_noise': 5., 'tau_theta': 1000., 'c': 10.}
        hand_coded_params_pop_excit_2 = {'preset_weights': weights_excit_2 }

        # Inhibitory
        params_pop_inhib_1 = {'tau_m': 10., 'tau_s': 6., 'Delta_noise': 5., 'tau_theta': 1000., 'c': 10.}
        hand_coded_params_pop_inhib_1 = {'preset_weights': weights_inhib_1 }
        params_pop_inhib_2 = {'tau_m': 10., 'tau_s': 6., 'Delta_noise': 5., 'tau_theta': 1000., 'c': 10.}
        hand_coded_params_pop_inhib_2 = {'preset_weights': weights_inhib_2 }

        params_pop_excit_1 = randomise_parameters(params_pop_excit_1, coeff=T(0.025), N_dim=pop_size)
        params_pop_excit_1 = zip_tensor_dicts(params_pop_excit_1, hand_coded_params_pop_excit_1)
        params_pop_excit_2 = randomise_parameters(params_pop_excit_2, coeff=T(0.025), N_dim=pop_size)
        params_pop_excit_2 = zip_tensor_dicts(params_pop_excit_2, hand_coded_params_pop_excit_2)
        params_pop_inhib_1 = randomise_parameters(params_pop_inhib_1, coeff=T(0.025), N_dim=pop_size)
        params_pop_inhib_1 = zip_tensor_dicts(params_pop_inhib_1, hand_coded_params_pop_inhib_1)
        params_pop_inhib_2 = randomise_parameters(params_pop_inhib_2, coeff=T(0.025), N_dim=pop_size)
        params_pop_inhib_2 = zip_tensor_dicts(params_pop_inhib_2, hand_coded_params_pop_inhib_2)

        randomised_params = zip_tensor_dicts(zip_tensor_dicts(zip_tensor_dicts(params_pop_excit_1, params_pop_excit_2), params_pop_inhib_1), params_pop_inhib_2)
    else:
        raise NotImplementedError("Model only supports 2 or 4 populations.")

    # TODO: Probabilities for spiking..?

    neuron_types = np.ones((N,))
    for i in range(int(N / 2)):
        neuron_types[-(1 + i)] = -1
    return microGIF(parameters=randomised_params, N=N, neuron_types=neuron_types)
