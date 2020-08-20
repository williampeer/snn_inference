import torch
import torch.nn as nn
from torch import tensor as T


class LIF_R(nn.Module):
    def __init__(self, device, parameters, tau_m=4.0, tau_g=2.0, v_rest=-65., N=10, w_mean=0.15, w_var=0.25,
                 pre_activation_coefficient=1.0, post_activation_coefficient=1.0):
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
        self.incr_theta_s = T(30.)
        self.N = N

        self.v = torch.zeros((self.N,))
        self.g = torch.zeros_like(self.v)  # syn. conductance
        self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step
        self.theta_s = T(30.) * torch.ones((self.N,))
        self.b_s = 0.5

        rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
        self.w = nn.Parameter(rand_ws, requires_grad=True)  # initialise with positive weights only
        self.v_rest = nn.Parameter(T(N * [v_rest]), requires_grad=True)
        # self.tau_m = nn.Parameter(T(N * [tau_m]), requires_grad=True)
        self.tau_g = nn.Parameter(T(N * [tau_g]), requires_grad=True)

        self.pre_activation_coefficient = T(pre_activation_coefficient)
        self.post_activation_coefficient = T(post_activation_coefficient)

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.g = self.g.clone().detach()
        self.spiked = self.spiked.clone().detach()

    def forward(self, x_in):
        # pre_act_in = self.pre_activation_coefficient * torch.add(self.w.matmul(self.g), x_in)
        # I = self.post_activation_coefficient * torch.sigmoid(pre_act_in)
        I = self.w.matmul(self.g) + x_in

        ones_g = torch.ones_like(self.tau_g)
        self.g = (ones_g - ones_g/self.tau_g) * self.g  # g = g - g/tau_g
        theta_s_next = (1-self.b_s) * self.theta_s

        C_m = 0.05
        dv = self.v_rest - self.v + I/C_m
        v_next = self.v + dv

        # differentiability
        self.spiked = torch.sigmoid(torch.sub(v_next, self.theta_s))

        # "filters"
        spiked = (v_next >= self.theta_s).float()  # thresholding when spiked isn't use for grad.s (non-differentiable)
        not_spiked = (spiked - 1.) / -1.  # flips the boolean mat.

        f_v = 0.15; delta_V = 12.
        v_reset = self.v_rest + f_v * (self.v - self.v_rest) - delta_V
        self.v = spiked * v_reset + not_spiked * v_next
        self.theta_s = spiked * (self.theta_s + self.incr_theta_s) + not_spiked * theta_s_next
        self.g = spiked * torch.ones_like(self.g) + not_spiked * self.g

        return self.v, self.spiked

