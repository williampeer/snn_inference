import torch
import torch.nn as nn
from torch import tensor as T


class LIF_ASC(nn.Module):
    def __init__(self, device, parameters, C_m=1.0, G=0.7, E_L=-65., N=10, w_mean=0.15, w_var=0.25,
                 delta_theta_s=30., b_s=0.5, R_I=25., f_v=0.15, delta_V=12., f_I=0.5, I_A=1.):
        super(LIF_ASC, self).__init__()
        # self.device = device

        if parameters:
            for key in parameters.keys():
                if key == 'C_m':
                    C_m = float(parameters[key])
                elif key == 'E_L':
                    E_L = float(parameters[key])
                elif key == 'G':
                    G = float(parameters[key])
                elif key == 'R_I':
                    R_I = float(parameters[key])
                elif key == 'N':
                    N = int(parameters[key])
                elif key == 'w_mean':
                    w_mean = float(parameters[key])
                elif key == 'w_var':
                    w_var = float(parameters[key])

        __constants__ = ['N']
        self.delta_theta_s = T(delta_theta_s)
        self.N = N

        self.v = torch.zeros((self.N,))
        self.g = torch.zeros_like(self.v)  # syn. conductance
        self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step
        self.theta_s = T(30.) * torch.ones((self.N,))
        self.b_s = T(b_s)
        self.I_additive = torch.zeros((self.N,))

        rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
        self.w = nn.Parameter(rand_ws, requires_grad=True)  # initialise with positive weights only
        self.E_L = nn.Parameter(T(N * [E_L]), requires_grad=True)
        self.C_m = nn.Parameter(T(N * [C_m]), requires_grad=True)
        self.G = nn.Parameter(T(N * [G]), requires_grad=True)
        self.f_I = nn.Parameter(T(N * [f_I]), requires_grad=True)
        self.R_I = nn.Parameter(T(N * [R_I]), requires_grad=True)

        self.f_v = T(f_v)
        self.delta_V = T(delta_V)
        self.I_A = T(I_A)

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.g = self.g.clone().detach()
        self.spiked = self.spiked.clone().detach()

        self.theta_s = self.theta_s.clone().detach()
        self.I_additive = self.I_additive.clone().detach()

        # self.f_I = self.f_I.clone().detach()
        # self.R_I = self.R_I.clone().detach()

    def forward(self, x_in):
        I = self.w.matmul(self.I_additive) + x_in
        dv = (self.G * (self.E_L - self.v) + I * self.R_I) / self.C_m
        v_next = self.v + dv

        # differentiable
        self.spiked = torch.sigmoid(torch.sub(v_next, self.theta_s))
        # non-differentiable, hard threshold
        spiked = (v_next >= self.theta_s).float()
        not_spiked = (spiked - 1.) / -1.

        self.v = torch.add(spiked * self.E_L, not_spiked * v_next)

        theta_s_next = (1 - self.b_s) * self.theta_s
        self.theta_s = spiked * (self.theta_s + self.delta_theta_s) + not_spiked * theta_s_next

        I_additive_decayed = (torch.ones_like(self.f_I) - self.f_I) * self.I_additive
        self.I_additive = spiked * (self.I_additive + self.I_A) + not_spiked * I_additive_decayed

        return self.v, self.spiked
