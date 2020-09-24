import torch
import torch.nn as nn
from torch import tensor as T

from Models.GLIF import GLIF
from experiments import randomise_parameters


def glif_recurrent_net(N = 12):
    glif_model = GLIF(device='cpu', parameters={}, N=N)
    w = torch.ones(N,1) * T(N * [0.2])  # + 0.2 * I
    glif_model.w = nn.Parameter(w, requires_grad=False)

    return glif_model


def random_glif_model(N = 12):
    torch.manual_seed(1234)
    free_parameters = {'w_mean': 0.2, 'w_var': 0.4, 'C_m': 1.5, 'G': 1.0, 'R_I': 19., 'E_L': -55., 'delta_theta_s': 25.,
                       'b_s': 0.4, 'f_v': 0.14, 'delta_V': 12., 'f_I': 0.4, 'I_A': 1., 'b_v': 0.5, 'a_v': 0.5, 'theta_inf': -25.}
    randomised_params = randomise_parameters(free_parameters, coeff=T(0.2))

    return GLIF(device='cpu', parameters=randomised_params, N=N)
