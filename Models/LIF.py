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

        self.R_I = R_I
        # self.to(self.device)

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.g = self.g.clone().detach()
        self.spiked = self.spiked.clone().detach()

    def forward(self, x_in):
        I = self.w.matmul(self.g) + x_in

        dv = (self.v_rest - self.v + I * self.R_I)/self.tau_m
        v_next = torch.add(self.v, dv)

        # differentiable soft threshold
        self.spiked = torch.sigmoid(torch.sub(v_next, self.spike_threshold))
        # non-differentiable, hard threshold
        spiked = (v_next >= self.spike_threshold).float()
        not_spiked = (spiked - 1.) / -1.

        # v = spiked * v_rest + not_spiked * v
        self.v = torch.add(spiked * self.v_rest, not_spiked * v_next)
        dg = -torch.div(self.g, self.tau_g)
        self.g = torch.add(spiked * torch.ones_like(self.g), not_spiked * torch.add(self.g, dg))

        return self.v, self.spiked


class LIF_complex(nn.Module):
    def __init__(self, device, parameters, tau_m=4.0, tau_g=2.0, v_rest=-65., N=10, w_mean=0.15, w_var=0.25, R_I=42.):
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
        self.R_I = nn.Parameter(T(N * [R_I]), requires_grad=True)

        # self.to(self.device)

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.g = self.g.clone().detach()
        self.spiked = self.spiked.clone().detach()

    def forward(self, x_in):
        I = self.w.matmul(self.g) + x_in

        dv = (self.v_rest - self.v + I * self.R_I) / self.tau_m
        v_next = torch.add(self.v, dv)

        # differentiable soft threshold
        self.spiked = torch.sigmoid(torch.sub(v_next, self.spike_threshold))
        # non-differentiable, hard threshold
        spiked = (v_next >= self.spike_threshold).float()
        not_spiked = (spiked - 1.) / -1.

        # v = spiked * v_rest + not_spiked * v
        self.v = torch.add(spiked * self.v_rest, not_spiked * v_next)
        dg = -torch.div(self.g, self.tau_g)
        self.g = torch.add(spiked * torch.ones_like(self.g), not_spiked * torch.add(self.g, dg))

        return self.v, self.spiked
