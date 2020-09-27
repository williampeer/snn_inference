import numpy as np
import torch
from torch import tensor as T

from Models.GLIF import GLIF
from experiments import randomise_parameters, zip_dicts


def glif1(N = 12):
    torch.manual_seed(1234)
    free_parameters = {'w_mean': 0.2, 'w_var': 0.4, 'C_m': 1.35, 'G': 0.75, 'R_I': 18., 'E_L': -58., 'delta_theta_s': 25.,
                       'b_s': 0.4, 'f_v': 0.14, 'delta_V': 12., 'f_I': 0.4, 'I_A': 1., 'b_v': 0.5, 'a_v': 0.5, 'theta_inf': -25.}
    randomised_params = randomise_parameters(free_parameters, coeff=T(0.2))
    # print('randomised_params', randomised_params)

    return GLIF(device='cpu', parameters=randomised_params, N=N)


def glif2(N = 12):  # CHAOTIC
    torch.manual_seed(1234)
    np.random.seed(1234)

    free_parameters = {'w_mean': 0.1, 'w_var': 0.5, 'delta_theta_s': 25., 'b_s': 0.4, 'f_v': 0.3, 'delta_V': 12.,
                       'b_v': 0.4, 'a_v': 0.4, 'theta_inf': -18., 'E_L': -57., 'C_m': 1.35, 'G': 0.7, 'I_A': 2.5, 'f_I': 0.3}
    randomised_params = randomise_parameters(free_parameters, coeff=T(0.1), N_dim=N)
    perturb_less_dict = {'R_I': 18.}
    randomised_R_I = randomise_parameters(perturb_less_dict, coeff=T(0.02), N_dim=N)
    randomised_params = zip_dicts(randomised_params, randomised_R_I)
    print('randomised_params', randomised_params)

    return GLIF(device='cpu', parameters=randomised_params, N=N)


def glif3(N = 12):  # CHAOTIC
    torch.manual_seed(1234)
    np.random.seed(1234)
    free_parameters = {'w_mean': 0.13, 'w_var': 0.33, 'C_m': 1.14, 'G': 0.82, 'R_I': 17., 'E_L': -60., 'delta_theta_s': 15.,
                       'b_s': 0.4, 'f_v': 0.3, 'delta_V': 10., 'f_I': 0.22, 'I_A': 2.5, 'b_v': 0.4, 'a_v': 0.35, 'theta_inf': -17.}
    randomised_params = randomise_parameters(free_parameters, coeff=T(0.1), N_dim=N)
    print('randomised_params', randomised_params)

    return GLIF(device='cpu', parameters=randomised_params, N=N)


def glif_slower_rate_async(N = 12):  # some silent neurons.
    torch.manual_seed(1234)
    np.random.seed(1234)
    free_parameters = {'w_mean': 0.3, 'w_var': 0.5, 'C_m': 1.5, 'G': 0.7, 'R_I': 18., 'E_L': -65., 'delta_theta_s': 12.,
                       'b_s': 0.4, 'f_v': 0.3, 'delta_V': 10., 'f_I': 0.25, 'I_A': 1.7, 'b_v': 0.4, 'a_v': 0.35, 'theta_inf': -20.}
    randomised_params = randomise_parameters(free_parameters, coeff=T(0.1), N_dim=N)
    print('randomised_params', randomised_params)

    return GLIF(device='cpu', parameters=randomised_params, N=N)


def glif_slower_more_synchronous(N = 12):
    torch.manual_seed(1234)
    np.random.seed(1234)
    free_parameters = {'w_mean': 0.2, 'w_var': 0.1, 'C_m': 2.7, 'G': 0.75, 'R_I': 15., 'E_L': -43., 'delta_theta_s': 27.,
                       'b_s': 0.2, 'f_v': 0.2, 'delta_V': 8., 'f_I': 0.6, 'I_A': 1.4, 'b_v': 0.1, 'a_v': 0.1, 'theta_inf': -26.}
    randomised_params = randomise_parameters(free_parameters, coeff=T(0.02), N_dim=N)
    print('randomised_params', randomised_params)

    return GLIF(device='cpu', parameters=randomised_params, N=N)
