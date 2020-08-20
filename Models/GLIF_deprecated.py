import torch
import torch.nn as nn
from torch import tensor as T


class GLIF_deprecated(nn.Module):
    def __init__(self, device, parameters, tau_m=4.0, tau_g=2.0, v_rest=-65., N=10, w_mean=0.15, w_var=0.25):
        super(GLIF_deprecated, self).__init__()
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
        # self.spike_threshold = T(30.)
        self.N = N

        # self.tau_m = T(tau_m)
        # self.tau_g = T(tau_g)
        # self.v_rest = T(v_rest)

        self.v = torch.zeros((self.N,))
        self.g = torch.zeros_like(self.v)  # syn. conductance
        self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step

        rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
        self.w = nn.Parameter(rand_ws, requires_grad=True)  # initialise with positive weights only
        self.v_rest = nn.Parameter(T(N * [v_rest]), requires_grad=True)
        self.tau_m = nn.Parameter(T(N * [tau_m]), requires_grad=True)
        self.tau_g = nn.Parameter(T(N * [tau_g]), requires_grad=True)

        self.b_s = nn.Parameter(T(N * [1.0]), requires_grad=False)

        self.a_v = nn.Parameter(T(N * [1.0]), requires_grad=False)
        self.b_v = nn.Parameter(T(N * [1.0]), requires_grad=False)

        self.theta_inf = T(0.)
        self.theta_v = T(0.) * torch.ones((self.N,))
        self.theta_s_reset = T(30.)
        self.theta_s = T(30.) * torch.ones((self.N,))

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.g = self.g.clone().detach()
        self.spiked = self.spiked.clone().detach()

    def forward(self, x_in):
        I = torch.sigmoid(torch.add(self.w.matmul(self.g), x_in))

        dv = torch.div(torch.add(torch.sub(self.v_rest, self.v), I), self.tau_m)
        self.v = torch.add(self.v, dv)

        # differentiability
        self.spiked = torch.sigmoid(torch.sub(self.v, self.theta_s + self.theta_v))

        # "filters"
        spiked = (self.v >= self.theta_s + self.theta_v).float()  # thresholding when spiked isn't use for grad.s (non-differentiable)
        not_spiked = (spiked - 1.) / -1.  # flips the boolean mat.

        self.v = torch.add(spiked * self.v_rest, not_spiked * self.v)
        dg = -torch.div(self.g, self.tau_g)
        self.g = torch.add(spiked * torch.ones_like(self.g),
                           not_spiked * torch.add(self.g, dg))

        # GLIF parameters
        dtheta_s = - self.b_s * self.theta_s
        dtheta_v = self.a_v * (self.v - self.v_rest) - self.b_v * (self.theta_v - self.theta_inf)
        self.theta_s = spiked * self.theta_s_reset + not_spiked * (self.theta_s + dtheta_s)
        self.theta_v = spiked * self.theta_v + not_spiked * (self.theta_v + dtheta_v)

        return self.v, self.spiked

