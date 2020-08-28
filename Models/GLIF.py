import torch
import torch.nn as nn
from torch import tensor as T


class GLIF(nn.Module):
    def __init__(self, device, parameters, tau_m=4.0, tau_g=2.0, v_rest=-65., N=10, w_mean=0.15, w_var=0.25,
                 delta_theta_s=30., b_s=0.5, R_I=12., f_v=0.15, delta_V=12., k_I_l=0.5, I_A=1., b_v=0.5, a_v=0.5,
                 theta_inf=-40):
        super(GLIF, self).__init__()
        # self.device = device

        if parameters:
            for key in parameters.keys():
                if key == 'tau_m':
                    tau_m = float(parameters[key])
                elif key == 'v_rest':
                    v_rest = float(parameters[key])
                elif key == 'N':
                    N = int(parameters[key])
                elif key == 'w_mean':
                    w_mean = float(parameters[key])
                elif key == 'w_var':
                    w_var = float(parameters[key])
                elif key == 'R_I':
                    R_I = float(parameters[key])

        __constants__ = ['N', 'v_rest', 'delta_theta_s', 'b_s', 'a_v', 'b_v', 'theta_inf']
        self.N = N
        self.v_rest = T(N * [v_rest])

        self.delta_theta_s = T(delta_theta_s)
        self.b_s = T(b_s)
        self.a_v = T(a_v)
        self.b_v = T(b_v)
        self.theta_inf = T(theta_inf)

        self.v = torch.zeros((self.N,))
        self.g = torch.zeros_like(self.v)  # syn. conductance
        self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step
        self.theta_s = T(30.) * torch.ones((self.N,))
        self.theta_v = torch.ones((self.N,))
        self.I_additive = torch.zeros((self.N,))

        rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
        self.w = nn.Parameter(rand_ws, requires_grad=True)  # initialise with positive weights only
        # self.v_rest = nn.Parameter(T(N * [v_rest]), requires_grad=True)
        self.tau_m = nn.Parameter(T(N * [tau_m]), requires_grad=True)
        # self.tau_m = T(N * [tau_m])
        # self.k_I_l = nn.Parameter(T(N * [k_I_l]), requires_grad=True)
        self.k_I_l = T(k_I_l)
        self.R_I = nn.Parameter(T(N * [R_I]), requires_grad=True)
        # self.R_I = T(R_I)
        # self.f_v = nn.Parameter(T(N * [f_v]), requires_grad=True)
        self.f_v = T(f_v)

        self.delta_V = T(delta_V)
        self.I_A = T(I_A)

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.g = self.g.clone().detach()
        self.spiked = self.spiked.clone().detach()

        self.theta_s = self.theta_s.clone().detach()
        self.theta_v = self.theta_v.clone().detach()
        self.I_additive = self.I_additive.clone().detach()

        # self.k_I_l = self.k_I_l.clone().detach()
        # self.R_I = self.R_I.clone().detach()

    def forward(self, x_in):
        I = x_in + self.w.matmul(self.I_additive)

        dv = (self.v_rest - self.v + I * self.R_I) / self.tau_m
        v_next = self.v + dv

        # differentiable
        self.spiked = torch.sigmoid(torch.sub(v_next, (self.theta_s + self.theta_v)))
        # NB: Non-differentiable, not used for gradients
        spiked = (v_next >= (self.theta_s + self.theta_v)).float()
        not_spiked = (spiked - 1.) / -1.

        v_reset = self.v_rest + self.f_v * (self.v - self.v_rest) - self.delta_V
        self.v = spiked * v_reset + not_spiked * v_next  # spike reset

        self.theta_s = (1 - self.b_s) * self.theta_s + spiked * self.delta_theta_s  # always decay
        d_theta_v = self.a_v * (self.v - self.v_rest) - self.b_v * (self.theta_v - self.theta_inf)
        self.theta_v = self.theta_v + not_spiked * d_theta_v

        self.I_additive = (torch.ones_like(self.k_I_l) - self.k_I_l) * self.I_additive \
                          + self.spiked * self.I_A

        return self.v, self.spiked
