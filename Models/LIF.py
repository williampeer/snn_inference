import torch
import torch.nn as nn
from torch import tensor as T


class LIF(nn.Module):
    def __init__(self, device, parameters, tau_m=1.0, tau_g=2.0, v_rest=-65., N=10, w_mean=0.15, w_var=0.25, R_I=42.):
        super(LIF, self).__init__()
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
                elif key == 'R_I':
                    R_I = float(parameters[key])

        __constants__ = ['spike_threshold', 'N']
        self.spike_threshold = T(30.)
        self.N = N

        self.v = torch.zeros((self.N,))
        self.g = torch.zeros_like(self.v)  # syn. conductance
        self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step

        rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
        self.w = nn.Parameter(rand_ws, requires_grad=True)  # initialise with positive weights only
        self.v_rest = nn.Parameter(T(N * [v_rest]), requires_grad=True)
        self.tau_m = nn.Parameter(T(N * [tau_m]), requires_grad=True)
        self.tau_g = nn.Parameter(T(N * [tau_g]), requires_grad=True)

        # constant chosen so as to enable spiking for this model
        self.R_I = R_I
        # self.to(self.device)

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.g = self.g.clone().detach()
        self.spiked = self.spiked.clone().detach()

    def forward(self, x_in):
        I = self.w.matmul(self.g) + x_in

        dv = (self.v_rest - self.v + I * self.R_I)/self.tau_m
        self.v = torch.add(self.v, dv)

        self.spiked = torch.sigmoid(torch.sub(self.v, self.spike_threshold))

        spiked = (self.v >= self.spike_threshold).float()  # thresholding when spiked isn't use for grad.s (non-differentiable)
        not_spiked = (spiked - 1.) / -1.  # flips the boolean mat.

        # v = spiked * v_rest + not_spiked * v
        self.v = torch.add(spiked * self.v_rest, not_spiked * self.v)
        dg = -torch.div(self.g, self.tau_g)
        self.g = torch.add(spiked * torch.ones_like(self.g),
                           not_spiked * torch.add(self.g, dg))

        return self.v, self.spiked


class LIF_complex(nn.Module):
    def __init__(self, device, parameters, tau_m=4.0, tau_g=2.0, v_rest=-65., N=10, w_mean=0.15, w_var=0.25,
                 pre_activation_coefficient=4.0, post_activation_coefficient=130.0):
        super(LIF_complex, self).__init__()
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
                elif key == 'pre_activation_coefficient':
                    pre_activation_coefficient = float(parameters[key])
                elif key == 'post_activation_coefficient':
                    post_activation_coefficient = float(parameters[key])

        __constants__ = ['spike_threshold', 'N']
        self.spike_threshold = T(30.)
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

        # self.pre_activation_coefficient = T(pre_activation_coefficient)
        # self.post_activation_coefficient = T(post_activation_coefficient)
        self.pre_activation_coefficient = nn.Parameter(T(pre_activation_coefficient), requires_grad=True)
        self.post_activation_coefficient = nn.Parameter(T(post_activation_coefficient), requires_grad=True)

        # self.to(self.device)

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.g = self.g.clone().detach()
        self.spiked = self.spiked.clone().detach()

    def forward(self, x_in):
        # I = w g + w x = w(g + x)
        pre_act_in = self.pre_activation_coefficient * torch.add(self.w.matmul(self.g), x_in)
        I = self.post_activation_coefficient * torch.sigmoid(pre_act_in)

        dg = -torch.div(self.g, self.tau_g)
        self.g = self.g + dg

        # dv = (v_rest - v + I) / tau_m
        dv = torch.div(torch.add(torch.sub(self.v_rest, self.v), I), self.tau_m)
        self.v = torch.add(self.v, dv)

        # Test for spikes
        self.spiked = torch.sigmoid(torch.sub(self.v, self.spike_threshold))

        spiked = (self.v >= self.spike_threshold).float()  # thresholding when spiked isn't use for grad.s (non-differentiable)
        not_spiked = (spiked - 1.) / -1.  # flips the boolean mat.

        # v = spiked * v_rest + not_spiked * v
        self.v = torch.add(spiked * self.v_rest, not_spiked * self.v)
        self.g = torch.add(spiked * torch.ones_like(self.g), not_spiked * self.g)

        return self.v, self.spiked
