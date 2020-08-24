import torch
import torch.nn as nn
from torch import tensor as T


class LIF_R(nn.Module):
    def __init__(self, device, parameters, tau_m=1.0, tau_g=2.0, v_rest=-65., N=10, w_mean=0.15, w_var=0.25,
                 delta_theta_s=30., b_s=0.5, R_I=20., f_v=0.15, delta_V=12.):
        super(LIF_R, self).__init__()
        # self.device = device

        if parameters:
            for key in parameters.keys():
                if key == 'tau_m':
                    tau_m = float(parameters[key])
                elif key == 'tau_g':
                    tau_g = float(parameters[key])
                elif key == 'v_rest':
                    v_rest = float(parameters[key])
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

        rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
        self.w = nn.Parameter(rand_ws, requires_grad=True)  # initialise with positive weights only
        self.v_rest = nn.Parameter(T(N * [v_rest]), requires_grad=True)
        self.tau_m = nn.Parameter(T(N * [tau_m]), requires_grad=True)
        self.tau_g = nn.Parameter(T(N * [tau_g]), requires_grad=True)

        self.R_I = T(R_I)
        self.f_v = T(f_v)  # Sample values: f_v = 0.15; delta_V = 12.
        self.delta_V = T(delta_V)

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.g = self.g.clone().detach()
        self.theta_s = self.theta_s.clone().detach()
        self.spiked = self.spiked.clone().detach()

    def forward(self, x_in):
        I = self.w.matmul(self.g) + x_in
        dv = (self.v_rest - self.v + I * self.R_I) / self.tau_m
        v_next = self.v + dv

        # differentiability
        self.spiked = torch.sigmoid(torch.sub(v_next, self.theta_s))
        # non-differentiable, hard threshold
        spiked = (v_next >= self.theta_s).float()
        not_spiked = (spiked - 1.) / -1.

        v_reset = self.v_rest + self.f_v * (self.v - self.v_rest) - self.delta_V
        self.v = spiked * v_reset + not_spiked * v_next

        theta_s_next = (1-self.b_s) * self.theta_s
        self.theta_s = spiked * (self.theta_s + self.delta_theta_s) + not_spiked * theta_s_next

        self.g = self.g - not_spiked * self.g/self.tau_g

        return self.v, self.spiked

