import torch
import torch.nn as nn
from torch import tensor as T

from Models.GLIF import GLIF
# from Models.Izhikevich import IzhikevichStable
# from Models.LIF import LIF


def glif_sleep_model():
    w_Wake = torch.ones((4, 1)) * torch.cat([T(4*[0.]), T(4*[-.4]), T(4*[-.2])])
    w_REM = torch.ones((4, 1)) * torch.cat([T(4*[.1]), T(4*[.16]), T(4*[0.])])
    w_NREM = torch.ones((4, 1)) * torch.cat([T(4*[-.168]), T(4*[-.13]), T(4*[0.])])
    w = torch.cat([w_Wake, w_REM, w_NREM])

    glif_model = GLIF(parameters={}, N=w.shape[0])
    glif_model.w = nn.Parameter(w, requires_grad=False)

    # wake, rem, nrem
    # tau_m = torch.cat([T(4 * [2.5]), T(4 * [1.1]), T(4 * [1.75])])
    # TODO: Sanity check of parameter values wrt. biological interpretation.
    #  Also: Changing one population results in different behaviour due to weights matrix (which is also negative).
    #  Also: Consider using (to get a spread of all parameters that will be fitted) all parameters (f_v and G)
    C_m = torch.cat([T(4 * [3.4]), T(4 * [1.4]), T(4 * [1.1])])
    E_L = torch.cat([T(4 * [-37.]), T(4 * [-52.]), T(4 * [-65.])])
    f_I = torch.cat([T(4 * [0.5]), T(4 * [0.4]), T(4 * [0.2])])

    glif_model.E_L = nn.Parameter(E_L, requires_grad=False)
    glif_model.C_m = nn.Parameter(C_m, requires_grad=False)
    glif_model.f_I = nn.Parameter(f_I, requires_grad=True)

    return glif_model


# def lif_sleep_model():
#     w_Wake = torch.ones((4, 1)) * torch.cat([T(4*[0.]), T(4*[-.4]), T(4*[-.2])])
#     w_REM = torch.ones((4, 1)) * torch.cat([T(4*[.1]), T(4*[.16]), T(4*[0.])])
#     w_NREM = torch.ones((4, 1)) * torch.cat([T(4*[-.168]), T(4*[-.13]), T(4*[0.])])
#     w = torch.cat([w_Wake, w_REM, w_NREM])
#
#     lif_model = LIF(device='cpu', parameters={}, N=w.shape[0], R_I=42.)
#     lif_model.w = nn.Parameter(w, requires_grad=False)
#
#     # wake, rem, nrem
#     # tau_m = torch.cat([T(4 * [2.5]), T(4 * [1.1]), T(4 * [1.75])])
#     C_m = torch.cat([T(4 * [2.5]), T(4 * [1.1]), T(4 * [1.75])])
#     tau_g = torch.cat([T(4 * [2.2]), T(4 * [1.5]), T(4 * [2.])])
#     E_L = torch.cat([T(4 * [-45.]), T(4 * [-72.]), T(4 * [-37.])])
#
#     lif_model.E_L = nn.Parameter(E_L, requires_grad=False)
#     lif_model.C_m = nn.Parameter(C_m, requires_grad=False)
#     lif_model.tau_g = nn.Parameter(tau_g, requires_grad=False)
#
#     return lif_model
#
#
# def izhikevich_sleep_model():
#     # w_Wake = torch.ones((4, 1)) * torch.cat([T(4 * [0.]), T(4 * [-.5]), T(4 * [-.25])])
#     # w_REM = torch.ones((4, 1)) * torch.cat([T(4 * [0.125]), T(4 * [0.6]), T(4 * [0.])])
#     # w_NREM = torch.ones((4, 1)) * torch.cat([T(4 * [-.21]), T(4 * [-.2125]), T(4 * [0.])])
#     w_Wake = torch.ones((4, 1)) * torch.cat([T(4 * [0.]), T(4 * [-.4]), T(4 * [-.2])])
#     w_REM = torch.ones((4, 1)) * torch.cat([T(4 * [.1]), T(4*[.16]), T(4 * [0.])])
#     w_NREM = torch.ones((4, 1)) * torch.cat([T(4 * [-.168]), T(4 * [-.13]), T(4 * [0.])])
#     w = torch.cat([w_Wake, w_REM, w_NREM])
#
#     m = IzhikevichStable(device='cpu', parameters={}, N=w.shape[0], R_I=0.8)
#     m.w = nn.Parameter(w, requires_grad=False)
#
#     a = torch.cat([T(4*[0.15]), T(4*[0.12]), T(4*[0.1])])
#     m.a = nn.Parameter(a, requires_grad=False)
#     b = torch.cat([T(4*[0.25]), T(4*[0.245]), T(4*[0.25])])
#     m.b = nn.Parameter(b, requires_grad=False)
#     c = torch.cat([T(4*[-65.]), T(4*[-44.]), T(4*[-62.])])
#     m.c = nn.Parameter(c, requires_grad=False)
#     d = torch.cat([T(4*[2.]), T(4*[1.3]), T(4*[2.])])
#     m.d = nn.Parameter(d, requires_grad=False)
#
#     tau_g = torch.cat([T(4*[5.]), T(4*[1.2]), T(4*[3.0])])  # synaptic conductance decay constant
#     # tau_g = torch.cat([T(4*[1.]), T(4*[1.0]), T(4*[1.0])])  # synaptic conductance decay constant
#     m.tau_g = nn.Parameter(tau_g, requires_grad=False)
#
#     return m
