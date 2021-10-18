import numpy as np
import torch
from torch import tensor as T

from Models.LowerDim.GLIF_soft_lower_dim import GLIF_soft_lower_dim
from Models.no_grad.GLIF_soft_no_grad import GLIF_soft_no_grad
from experiments import randomise_parameters, zip_tensor_dicts


def glif_soft_continuous_ensembles_model_dales_compliant(random_seed, pop_size=1, N_pops=2):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    weights_std = 0.05
    # weights_std = 0

    N = N_pops * pop_size

    if N_pops == 2:
        weights_excit_1 = (torch.ones((pop_size, 1)) + (2 * weights_std * torch.randn((pop_size, N))) - weights_std) * \
                          torch.cat([T(pop_size * [0.2]), T(pop_size * [0.8])])
        weights_inhib_1 = (torch.ones((pop_size, 1)) + (2 * weights_std * torch.randn((pop_size, N))) - weights_std) * \
                          torch.cat([T(pop_size * [.9]), T(pop_size * [.4])])

        # Excitatory:
        params_pop_excit_1 = {'tau_m': 4., 'G': 0.6, 'E_L': -58., 'delta_theta_s': 18., 'b_s': 0.35, 'f_v': 0.14,
                              'delta_V': 10.,
                              'f_I': 0.45, 'b_v': 0.3, 'a_v': 0.2, 'theta_inf': -5., 'tau_g': 4.5}
        hand_coded_params_pop_excit_1 = {'preset_weights': weights_excit_1}

        # Inhibitory
        params_pop_inhib_1 = {'tau_m': 2.5, 'G': 0.7, 'E_L': -66., 'delta_theta_s': 18., 'b_s': 0.25, 'f_v': 0.14,
                              'delta_V': 12.,
                              'f_I': 0.35, 'b_v': 0.4, 'a_v': 0.3, 'theta_inf': -10., 'tau_g': 3.}
        hand_coded_params_pop_inhib_1 = {'preset_weights': weights_inhib_1}

        params_pop_excit_1 = randomise_parameters(params_pop_excit_1, coeff=T(0.025), N_dim=pop_size)
        params_pop_excit_1 = zip_tensor_dicts(params_pop_excit_1, hand_coded_params_pop_excit_1)
        params_pop_inhib_1 = randomise_parameters(params_pop_inhib_1, coeff=T(0.025), N_dim=pop_size)
        params_pop_inhib_1 = zip_tensor_dicts(params_pop_inhib_1, hand_coded_params_pop_inhib_1)

        randomised_params = zip_tensor_dicts(params_pop_excit_1, params_pop_inhib_1)
    elif N_pops == 4:
        # up to 4 populations
        weights_excit_1 = (torch.ones((pop_size, 1)) + (
                    2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
            [T(pop_size * [0.6]), T(pop_size * [0.4]), T(pop_size * [0.9]), T(pop_size * [0.9])])
        weights_excit_2 = (torch.ones((pop_size, 1)) + (
                    2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
            [T(pop_size * [.4]), T(pop_size * [.6]), T(pop_size * [.9]), T(pop_size * [.9])])
        weights_inhib_1 = (torch.ones((pop_size, 1)) + (
                    2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
            [T(pop_size * [.8]), T(pop_size * [.8]), T(pop_size * [.2]), T(pop_size * [.1])])
        weights_inhib_2 = (torch.ones((pop_size, 1)) + (
                    2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
            [T(pop_size * [.9]), T(pop_size * [.9]), T(pop_size * [.1]), T(pop_size * [.2])])

        # Excitatory:
        params_pop_excit_1 = {'tau_m': 4., 'G': 0.7, 'E_L': -58., 'delta_theta_s': 18., 'b_s': 0.4, 'f_v': 0.14,
                              'delta_V': 12., 'f_I': 0.45, 'b_v': 0.3, 'a_v': 0.2, 'theta_inf': -4., 'tau_g': 4.5 }
        hand_coded_params_pop_excit_1 = {'preset_weights': weights_excit_1 }
        params_pop_excit_2 = {'tau_m': 4.9, 'G': 0.6, 'E_L': -55., 'delta_theta_s': 14., 'b_s': 0.3, 'f_v': 0.14,
                              'delta_V': 12., 'f_I': 0.4, 'b_v': 0.35, 'a_v': 0.25, 'theta_inf': -2., 'tau_g': 5.8 }
        hand_coded_params_pop_excit_2 = {'preset_weights': weights_excit_2 }

        # Inhibitory
        params_pop_inhib_1 = {'tau_m': 2.6, 'G': 0.7, 'E_L': -66., 'delta_theta_s': 18., 'b_s': 0.25, 'f_v': 0.14,
                              'delta_V': 12., 'f_I': 0.35, 'b_v': 0.4, 'a_v': 0.3, 'theta_inf': -11., 'tau_g': 3. }
        hand_coded_params_pop_inhib_1 = {'preset_weights': weights_inhib_1 }
        params_pop_inhib_2 = {'tau_m': 2.1, 'G': 0.8, 'E_L': -66., 'delta_theta_s': 18., 'b_s': 0.25, 'f_v': 0.14,
                              'delta_V': 12., 'f_I': 0.35, 'b_v': 0.4, 'a_v': 0.3, 'theta_inf': -7., 'tau_g': 2.6 }
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

    neuron_types = np.ones((N,))
    for i in range(int(N / 2)):
        neuron_types[-(1 + i)] = -1
    return GLIF_soft_no_grad(parameters=randomised_params, N=N, neuron_types=neuron_types)
    # return GLIF_soft_lower_dim(parameters=randomised_params, N=N, neuron_types=neuron_types)


def glif_soft_cortical_populations(random_seed, pop_size=1, N_pops=2):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    weights_std = 0.05
    # weights_std = 0

    N = N_pops * pop_size

    if N_pops == 2:
        weights_excit_1 = (torch.ones((pop_size, 1)) + (2 * weights_std * torch.randn((pop_size, N))) - weights_std) * \
                          torch.cat([T(pop_size * [0.2]), T(pop_size * [0.2])])
        weights_inhib_1 = (torch.ones((pop_size, 1)) + (2 * weights_std * torch.randn((pop_size, N))) - weights_std) * \
                          torch.cat([T(pop_size * [.9]), T(pop_size * [.4])])

        # Excitatory:
        params_pop_excit_1 = {'tau_m': 4., 'G': 0.6, 'E_L': -58., 'delta_theta_s': 18., 'b_s': 0.35, 'f_v': 0.14,
                              'delta_V': 10.,
                              'f_I': 0.45, 'b_v': 0.3, 'a_v': 0.2, 'theta_inf': -5., 'tau_g': 4.5}
        hand_coded_params_pop_excit_1 = {'preset_weights': weights_excit_1}

        # Inhibitory
        params_pop_inhib_1 = {'tau_m': 2.5, 'G': 0.7, 'E_L': -66., 'delta_theta_s': 18., 'b_s': 0.25, 'f_v': 0.14,
                              'delta_V': 12.,
                              'f_I': 0.35, 'b_v': 0.4, 'a_v': 0.3, 'theta_inf': -10., 'tau_g': 3.}
        hand_coded_params_pop_inhib_1 = {'preset_weights': weights_inhib_1}

        params_pop_excit_1 = randomise_parameters(params_pop_excit_1, coeff=T(0.025), N_dim=pop_size)
        params_pop_excit_1 = zip_tensor_dicts(params_pop_excit_1, hand_coded_params_pop_excit_1)
        params_pop_inhib_1 = randomise_parameters(params_pop_inhib_1, coeff=T(0.025), N_dim=pop_size)
        params_pop_inhib_1 = zip_tensor_dicts(params_pop_inhib_1, hand_coded_params_pop_inhib_1)

        randomised_params = zip_tensor_dicts(params_pop_excit_1, params_pop_inhib_1)
    elif N_pops == 4:
        # up to 4 populations
        weights_excit_1 = (torch.ones((pop_size, 1)) + (
                    2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
            [T(pop_size * [0.6]), T(pop_size * [0.4]), T(pop_size * [0.9]), T(pop_size * [0.9])])
        weights_excit_2 = (torch.ones((pop_size, 1)) + (
                    2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
            [T(pop_size * [.4]), T(pop_size * [.6]), T(pop_size * [.9]), T(pop_size * [.9])])
        weights_inhib_1 = (torch.ones((pop_size, 1)) + (
                    2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
            [T(pop_size * [.8]), T(pop_size * [.8]), T(pop_size * [.2]), T(pop_size * [.1])])
        weights_inhib_2 = (torch.ones((pop_size, 1)) + (
                    2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
            [T(pop_size * [.9]), T(pop_size * [.9]), T(pop_size * [.1]), T(pop_size * [.2])])

        # Excitatory:
        params_pop_excit_1 = {'tau_m': 4., 'G': 0.7, 'E_L': -52., 'delta_theta_s': 18., 'b_s': 0.4, 'f_v': 0.14,
                              'delta_V': 12., 'f_I': 0.45, 'b_v': 0.3, 'a_v': 0.2, 'theta_inf': -4., 'tau_g': 4.5 }
        hand_coded_params_pop_excit_1 = {'preset_weights': weights_excit_1 }
        params_pop_excit_2 = {'tau_m': 4.9, 'G': 0.6, 'E_L': -64., 'delta_theta_s': 14., 'b_s': 0.3, 'f_v': 0.14,
                              'delta_V': 12., 'f_I': 0.4, 'b_v': 0.35, 'a_v': 0.25, 'theta_inf': -2., 'tau_g': 5.8 }
        hand_coded_params_pop_excit_2 = {'preset_weights': weights_excit_2 }

        # Inhibitory
        params_pop_inhib_1 = {'tau_m': 2.6, 'G': 0.7, 'E_L': -66., 'delta_theta_s': 18., 'b_s': 0.25, 'f_v': 0.14,
                              'delta_V': 12., 'f_I': 0.35, 'b_v': 0.4, 'a_v': 0.3, 'theta_inf': -11., 'tau_g': 3. }
        hand_coded_params_pop_inhib_1 = {'preset_weights': weights_inhib_1 }
        params_pop_inhib_2 = {'tau_m': 2.1, 'G': 0.8, 'E_L': -66., 'delta_theta_s': 18., 'b_s': 0.25, 'f_v': 0.14,
                              'delta_V': 12., 'f_I': 0.35, 'b_v': 0.4, 'a_v': 0.3, 'theta_inf': -7., 'tau_g': 2.6 }
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

    neuron_types = np.ones((N,))
    for i in range(int(N / 2)):
        neuron_types[-(1 + i)] = -1
    return GLIF_soft_no_grad(parameters=randomised_params, N=N, neuron_types=neuron_types)


# def lif_r_soft_continuous_ensembles_model_dales_compliant(random_seed, N = 12):
#     torch.manual_seed(random_seed)
#     np.random.seed(random_seed)
#
#     pop_size = int(N / 3)
#     pop_size_last = N - 2*pop_size
#     # params_pop1 = {'tau_m': 2.5, 'tau_s': 3.0, 'G': 0.9, 'E_L': -39., 'delta_theta_s': 18., 'b_s': 0.4, 'f_v': 0.14, 'delta_V': 10.}
#     params_pop1 = {'tau_m': 6.2, 'E_L': -62., 'G': 0.85, 'delta_theta_s': 18., 'b_s': 0.4, 'f_v': 0.14, 'delta_V': 10., 'tau_g': 7.}
#     weights_std = 0.05
#     # weights_std = 0
#     hand_coded_params_pop1 = {'preset_weights': (torch.ones((pop_size, 1)) +
#                                                  (2 * weights_std * torch.randn((pop_size, N))) - weights_std) *
#                                                 # torch.cat([T(pop_size*[0.06]), T(pop_size*[0.03]), T(pop_size*[0.01])])}
#                                                 # torch.cat([T(pop_size * [0.2]), T(pop_size * [0.15]), T(pop_size * [0.05])])}
#                                                 # torch.cat([T(pop_size * [0.2]), T(pop_size * [0.1]), T(pop_size * [0.03])])}
#                                                 # torch.cat([T(pop_size * [0.0]), T(pop_size * [0.0]), T(pop_size * [0.0])])}
#                                                 torch.cat([T(pop_size * [0.1]), T(pop_size * [0.06]), T(pop_size_last * [0.06])])}
#
#     # params_pop2 = {'tau_m': 1.5, 'tau_s': 2.4, 'G': 0.8, 'E_L': -49., 'delta_theta_s': 14., 'b_s': 0.3, 'f_v': 0.14, 'delta_V': 12.}
#     params_pop2 = {'tau_m': 5.3, 'E_L': -66., 'G': 0.8, 'delta_theta_s': 14., 'b_s': 0.3, 'f_v': 0.14, 'delta_V': 12., 'tau_g': 4.5}
#     hand_coded_params_pop2 = {'preset_weights': (torch.ones((pop_size, 1)) +
#                                                  (2 * weights_std * torch.randn((pop_size, N))) - weights_std) *
#                                                 # torch.cat([T(4*[.02]), T(4*[.3]), T(4*[0.15])])}
#                                                 # torch.cat([T(pop_size * [.08]), T(pop_size * [.38]), T(pop_size * [0.32])])}
#                                                 # torch.cat([T(pop_size * [0.04]), T(pop_size * [0.4]), T(pop_size * [0.2])])}
#                                                 # torch.cat([T(pop_size * [0.0]), T(pop_size * [0.0]), T(pop_size * [0.0])])}
#                                                 torch.cat([T(pop_size * [.08]), T(pop_size * [.3]), T(pop_size_last * [.15])])}
#
#     # params_pop3 = {'tau_m': 1.16, 'tau_s': 1.8, 'G': 0.7, 'E_L': -61., 'delta_theta_s': 18., 'b_s': 0.25, 'f_v': 0.14, 'delta_V': 12.}
#     params_pop3 = {'tau_m': 5.1, 'E_L': -72., 'G': 0.8, 'delta_theta_s': 18., 'b_s': 0.25, 'f_v': 0.14, 'delta_V': 12., 'tau_g': 3.3}
#     hand_coded_params_pop3 = {'preset_weights': (torch.ones((pop_size_last, 1)) +
#                                                  (2 * weights_std * torch.randn((pop_size_last, N))) - weights_std) *
#                                                 # torch.cat([T(pop_size * [-.4]), T(pop_size * [-.2]), T(pop_size * [-0.02])])}
#                                                 # torch.cat([T(4*[-.3]), T(4*[-.28]), T(4*[-0.07])])}
#                                                 # torch.cat([T(pop_size * [-0.32]), T(pop_size * [-0.25]), T(pop_size * [-0.15])])}
#                                                 # torch.cat([T(pop_size * [0.0]), T(pop_size * [0.0]), T(pop_size * [0.0])])}
#                                                 torch.cat([T(pop_size * [.12]), T(pop_size * [.18]), T(pop_size_last * [.25])])}
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
#     return LIF_R_soft_no_grad(parameters=randomised_params, N=N, neuron_types=neuron_types)


# def lif_asc_continuous_ensembles_model_dales_compliant(random_seed, N = 12):
#     torch.manual_seed(random_seed)
#     np.random.seed(random_seed)
#
#     pop_size = int(N / 3)
#     pop_size_last = N - 2*pop_size
#     params_pop1 = {'tau_m': 6., 'E_L': -55., 'f_I': 0.45, 'G': 0.9, 'delta_theta_s': 18., 'b_s': 0.3, 'f_v': 0.14,
#                    'delta_V': 12., 'tau_s': 6.}
#     weights_std = 0.05
#     # weights_std = 0
#     hand_coded_params_pop1 = {'preset_weights': (torch.ones((pop_size, 1)) +
#                                                  (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
#                                                 # torch.cat([T(pop_size*[0.28]), T(pop_size*[0.2]), T(pop_size*[0.15])])}
#                                                 # torch.cat([T(pop_size * [0.2]), T(pop_size * [0.08]), T(pop_size * [0.03])])}
#                                                 # torch.cat([T(pop_size*[0.0]), T(pop_size*[0.0]), T(pop_size*[0.0])])}
#                                                 torch.cat([T(pop_size * [0.15]), T(pop_size * [0.08]), T(pop_size_last * [0.05])])}
#
#     params_pop2 = {'tau_m': 4.5, 'E_L': -62., 'f_I': 0.4, 'G': 0.9, 'delta_theta_s': 16., 'b_s': 0.3, 'f_v': 0.14,
#                    'delta_V': 12., 'tau_s': 4.5}
#     hand_coded_params_pop2 = {'preset_weights': (torch.ones((pop_size, 1)) +
#                                                  (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
#                                                 # torch.cat([T(4*[.1]), T(4*[.43]), T(4*[0.34])])}
#                                                 # torch.cat([T(pop_size * [0.04]), T(pop_size * [0.36]), T(pop_size * [0.15])])}
#                                                 # torch.cat([T(4*[.0]), T(4*[.0]), T(4*[0.0])])}
#                                                 torch.cat([T(pop_size * [.08]), T(pop_size * [.35]), T(pop_size_last * [.25])])}
#
#     params_pop3 = {'tau_m': 4., 'E_L': -68., 'f_I': 0.35, 'G': 0.9, 'delta_theta_s': 18., 'b_s': 0.25, 'f_v': 0.14,
#                    'delta_V': 12., 'tau_s': 2.6}
#     hand_coded_params_pop3 = {'preset_weights': (torch.ones((pop_size_last, 1)) +
#                                                  (2*weights_std * torch.randn((pop_size_last, N))) - weights_std) *
#                                                 # torch.cat([T(4*[-.16]), T(4*[-.4]), T(4*[-0.12])])}
#                                                 # torch.cat([T(pop_size * [-.36]), T(pop_size * [-.22]), T(pop_size * [-0.08])])}
#                                                 # torch.cat([T(4*[-.0]), T(4*[-.0]), T(4*[-0.0])])}
#                                                 torch.cat([T(pop_size * [-.2]), T(pop_size * [-.3]), T(pop_size_last * [-.3])])}
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
#     return LIF_ASC(parameters=randomised_params, N=N, neuron_types=neuron_types)
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
#
#
# def glif_soft_continuous_ensembles_model_dales_compliant(random_seed, pop_size=1, N=2):
#     torch.manual_seed(random_seed)
#     np.random.seed(random_seed)
#
#     params_pop1 = {'tau_m': 3.4, 'G': 0.7, 'E_L': -52., 'delta_theta_s': 18., 'b_s': 0.4, 'f_v': 0.14, 'delta_V': 10.,
#                    'f_I': 0.5, 'b_v': 0.3, 'a_v': 0.2, 'theta_inf': -12., 'tau_g': 4.5}
#     weights_std = 0.05
#     # weights_std = 0
#     hand_coded_params_pop1 = {'preset_weights': (torch.ones((pop_size, 1)) +
#                                                  (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
#                                                 # torch.cat([T(pop_size*[0.06]), T(pop_size*[0.03]), T(pop_size*[0.01])])}
#                                                 # torch.cat([T(pop_size * [0.2]), T(pop_size * [0.15]), T(pop_size * [0.05])])}
#                                                 # torch.cat([T(pop_size * [0.0]), T(pop_size * [0.0]), T(pop_size * [0.0])])}
#                                                 torch.cat([T(pop_size * [0.14]), T(pop_size * [0.08]), T(pop_size * [0.06])])}
#
#     params_pop2 = {'tau_m': 2.9, 'G': 0.7, 'E_L': -58., 'delta_theta_s': 14., 'b_s': 0.3, 'f_v': 0.14, 'delta_V': 12.,
#                    'f_I': 0.4, 'b_v': 0.35, 'a_v': 0.25, 'theta_inf': -14., 'tau_g': 3.8}
#     hand_coded_params_pop2 = {'preset_weights': (torch.ones((pop_size, 1)) +
#                                                  (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
#                                                 # torch.cat([T(4*[.02]), T(4*[.3]), T(4*[0.15])])}
#                                                 # torch.cat([T(pop_size * [.08]), T(pop_size * [.38]), T(pop_size * [0.32])])}
#                                                 # torch.cat([T(pop_size * [0.0]), T(pop_size * [0.0]), T(pop_size * [0.0])])}
#                                                 torch.cat([T(pop_size * [.08]), T(pop_size * [.3]), T(pop_size * [.2])])}
#
#     params_pop3 = {'tau_m': 2.6, 'G': 0.8, 'E_L': -66., 'delta_theta_s': 18., 'b_s': 0.25, 'f_v': 0.14, 'delta_V': 12.,
#                    'f_I': 0.35, 'b_v': 0.4, 'a_v': 0.3, 'theta_inf': -11., 'tau_g': 2.4}
#     hand_coded_params_pop3 = {'preset_weights': (torch.ones((pop_size_last, 1)) +
#                                                  (2*weights_std * torch.randn((pop_size_last, N))) - weights_std) *
#                                                 # torch.cat([T(pop_size * [-.4]), T(pop_size * [-.2]), T(pop_size * [-0.02])])}
#                                                 # torch.cat([T(pop_size * [-.4]), T(pop_size * [-.2]), T(pop_size * [-0.02])])}
#                                                 # torch.cat([T(pop_size * [0.0]), T(pop_size * [0.0]), T(pop_size * [0.0])])}
#                                                 torch.cat([T(pop_size * [.06]), T(pop_size * [.1]), T(pop_size * [.1])])}
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
#     for i in range(int(N / 2)):
#         neuron_types[-(1 + i)] = -1
#     return GLIF_soft_no_grad(parameters=randomised_params, N=N, neuron_types=neuron_types)


# def glif_soft_positive_weights_continuous_ensembles_model_dales_compliant(random_seed, N = 12):
#     torch.manual_seed(random_seed)
#     np.random.seed(random_seed)
#
#     pop_size = int(N / 3)
#     pop_size_last = N - 2*pop_size
#     params_pop1 = {'tau_m': 3.4, 'G': 0.7, 'E_L': -52., 'delta_theta_s': 18., 'b_s': 0.4, 'f_v': 0.14, 'delta_V': 10.,
#                    'f_I': 0.5, 'b_v': 0.3, 'a_v': 0.2, 'theta_inf': -12., 'tau_g': 4.5}
#     weights_std = 0.05
#     # weights_std = 0
#     hand_coded_params_pop1 = {'preset_weights': (torch.ones((pop_size, 1)) +
#                                                  (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
#                                                 # torch.cat([T(pop_size*[0.06]), T(pop_size*[0.03]), T(pop_size*[0.01])])}
#                                                 # torch.cat([T(pop_size * [0.2]), T(pop_size * [0.15]), T(pop_size * [0.05])])}
#                                                 # torch.cat([T(pop_size * [0.0]), T(pop_size * [0.0]), T(pop_size * [0.0])])}
#                                                 torch.cat([T(pop_size * [0.14]), T(pop_size * [0.08]), T(pop_size_last * [0.06])])}
#
#     params_pop2 = {'tau_m': 2.9, 'G': 0.7, 'E_L': -58., 'delta_theta_s': 14., 'b_s': 0.3, 'f_v': 0.14, 'delta_V': 12.,
#                    'f_I': 0.4, 'b_v': 0.35, 'a_v': 0.25, 'theta_inf': -14., 'tau_g': 3.8}
#     hand_coded_params_pop2 = {'preset_weights': (torch.ones((pop_size, 1)) +
#                                                  (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
#                                                 # torch.cat([T(4*[.02]), T(4*[.3]), T(4*[0.15])])}
#                                                 # torch.cat([T(pop_size * [.08]), T(pop_size * [.38]), T(pop_size * [0.32])])}
#                                                 # torch.cat([T(pop_size * [0.0]), T(pop_size * [0.0]), T(pop_size * [0.0])])}
#                                                 torch.cat([T(pop_size * [.08]), T(pop_size * [.3]), T(pop_size_last * [.2])])}
#
#     params_pop3 = {'tau_m': 2.6, 'G': 0.8, 'E_L': -66., 'delta_theta_s': 18., 'b_s': 0.25, 'f_v': 0.14, 'delta_V': 12.,
#                    'f_I': 0.35, 'b_v': 0.4, 'a_v': 0.3, 'theta_inf': -11., 'tau_g': 2.4}
#     hand_coded_params_pop3 = {'preset_weights': (torch.ones((pop_size_last, 1)) +
#                                                  (2*weights_std * torch.randn((pop_size_last, N))) - weights_std) *
#                                                 # torch.cat([T(pop_size * [-.4]), T(pop_size * [-.2]), T(pop_size * [-0.02])])}
#                                                 # torch.cat([T(pop_size * [-.4]), T(pop_size * [-.2]), T(pop_size * [-0.02])])}
#                                                 # torch.cat([T(pop_size * [0.0]), T(pop_size * [0.0]), T(pop_size * [0.0])])}
#                                                 torch.cat([T(pop_size * [.06]), T(pop_size * [.1]), T(pop_size_last * [.1])])}
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
#     # for i in range(int(N / 3)):
#     #     neuron_types[-(1 + i)] = -1
#     return GLIF_soft_no_grad(parameters=randomised_params, N=N, neuron_types=neuron_types)
