import torch
import torch.nn as nn
from torch import tensor as T


class GLIF(nn.Module):
    parameter_names = ['w', 'E_L', 'C_m', 'G', 'R_I', 'f_v', 'f_I', 'delta_theta_s', 'b_s', 'a_v', 'b_v', 'theta_inf', 'delta_V', 'I_A']

    def __init__(self, device, parameters, C_m=1., G=0.7, R_I=18., E_L=-60., N=10, w_mean=0.55, w_var=0.65,
                 delta_theta_s=30., b_s=0.3, f_v=0.15, delta_V=12., f_I=0.3, I_A=1., b_v=0.5, a_v=0.5, theta_inf=-20):
        super(GLIF, self).__init__()
        # self.device = device

        if parameters:
            for key in parameters.keys():
                if key == 'C_m':
                    C_m = float(parameters[key])
                elif key == 'G':
                    G = float(parameters[key])
                elif key == 'R_I':
                    R_I = float(parameters[key])
                elif key == 'E_L':
                    E_L = float(parameters[key])
                elif key == 'delta_theta_s':
                    delta_theta_s = float(parameters[key])
                elif key == 'b_s':
                    b_s = float(parameters[key])
                elif key == 'f_v':
                    f_v = float(parameters[key])
                elif key == 'delta_V':
                    delta_V = float(parameters[key])
                elif key == 'f_I':
                    f_I = float(parameters[key])
                elif key == 'b_v':
                    b_v = float(parameters[key])
                elif key == 'a_v':
                    a_v = float(parameters[key])
                elif key == 'theta_inf':
                    theta_inf = float(parameters[key])
                elif key == 'N':
                    N = int(parameters[key])
                elif key == 'w_mean':
                    w_mean = float(parameters[key])
                elif key == 'w_var':
                    w_var = float(parameters[key])

        __constants__ = ['N', 'E_L', 'delta_theta_s', 'b_s', 'a_v', 'b_v', 'theta_inf']
        self.N = N
        # self.E_L = T(N * [E_L])

        # self.delta_theta_s = T(delta_theta_s)
        # self.b_s = T(b_s)
        # self.a_v = T(a_v)
        # self.b_v = T(b_v)
        # self.theta_inf = T(theta_inf)

        self.v = E_L * torch.ones((self.N,))
        self.g = torch.zeros_like(self.v)  # syn. conductance
        self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step
        self.theta_s = T(30.) * torch.ones((self.N,))
        self.theta_v = torch.ones((self.N,))
        self.I_additive = torch.zeros((self.N,))

        rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
        self.w = nn.Parameter(rand_ws, requires_grad=True)  # initialise with positive weights only
        self.E_L = nn.Parameter(T(N * [E_L]), requires_grad=True)
        self.C_m = nn.Parameter(T(N * [C_m]), requires_grad=True)
        self.G = nn.Parameter(T(N * [G]), requires_grad=True)
        self.R_I = nn.Parameter(T(N * [R_I]), requires_grad=True)
        self.f_v = nn.Parameter(T(N * [f_v]), requires_grad=True)
        self.f_I = nn.Parameter(T(N * [f_I]), requires_grad=True)

        self.delta_theta_s = nn.Parameter(T(N * [delta_theta_s]), requires_grad=True)
        self.b_s = nn.Parameter(T(N * [b_s]), requires_grad=True)
        self.a_v = nn.Parameter(T(N * [a_v]), requires_grad=True)
        self.b_v = nn.Parameter(T(N * [b_v]), requires_grad=True)
        self.theta_inf = nn.Parameter(T(N * [theta_inf]), requires_grad=True)
        # self.delta_V = T(delta_V)
        # self.I_A = T(I_A)
        self.delta_V = nn.Parameter(T(N * [delta_V]), requires_grad=True)
        self.I_A = nn.Parameter(T(N * [I_A]), requires_grad=True)

    def reset(self):
        self.reset_hidden_state()
        for p in self.parameters():
            if hasattr(p, 'reset_parameters'):
                p.reset_parameters()
                print('DEBUG: reset_parameters(), p: {}'.format(p))

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.g = self.g.clone().detach()
        self.spiked = self.spiked.clone().detach()
        self.theta_s = self.theta_s.clone().detach()
        self.theta_v = self.theta_v.clone().detach()
        self.I_additive = self.I_additive.clone().detach()

    def forward(self, x_in):
        I = x_in + self.w.matmul(self.I_additive)

        dv = (I * self.R_I - self.G * (self.v - self.E_L)) / self.C_m
        v_next = self.v + dv

        # differentiable
        self.spiked = torch.sigmoid(torch.sub(v_next, (self.theta_s + self.theta_v)))
        # NB: Non-differentiable, not used for gradients
        spiked = (v_next >= (self.theta_s + self.theta_v)).float()
        not_spiked = (spiked - 1.) / -1.

        v_reset = self.E_L + self.f_v * (self.v - self.E_L) - self.delta_V
        self.v = spiked * v_reset + not_spiked * v_next  # spike reset

        self.theta_s = (1. - self.b_s) * self.theta_s + spiked * self.delta_theta_s  # always decay
        d_theta_v = self.a_v * (self.v - self.E_L) - self.b_v * (self.theta_v - self.theta_inf)
        self.theta_v = self.theta_v + not_spiked * d_theta_v

        self.I_additive = (1. - self.f_I) * self.I_additive \
                          + self.spiked * self.I_A

        return self.v, self.spiked
