import torch
import torch.nn as nn
from torch import tensor as T


class LIF(nn.Module):
    def __init__(self):
        super(LIF, self).__init__()
        # self.device = device

        w_Wake = torch.ones((4, 1)) * torch.cat([T(4*[0.]), T(4*[-.4]), T(4*[-.2])])
        w_REM = torch.ones((4, 1)) * torch.cat([T(4*[.1]), T(4*[.16]), T(4*[0.])])
        w_NREM = torch.ones((4, 1)) * torch.cat([T(4*[-.168]), T(4*[-.13]), T(4*[0.])])
        w = torch.cat([w_Wake, w_REM, w_NREM])

        __constants__ = ['spike_threshold', 'N']
        self.spike_threshold = T(30.)
        self.N = 12

        # wake, rem, nrem
        tau_m = torch.cat([T(4*[2.5]), T(4*[0.6]), T(4*[1.75])])
        tau_g = torch.cat([T(4*[2.5]), T(4*[1.0]), T(4*[2.5])])
        v_rest = torch.cat([T(4*[-37.]), T(4*[-80.]), T(4*[-37.])])
        # v_rest = T(-65.)

        post_activation_coefficient = T(70.)
        # pre_activation_coefficient = T(2.0)
        pre_activation_coefficient = torch.cat([T(4*[6.0]), T(4*[2.0]), T(4*[3.])])

        self.w = nn.Parameter(w, requires_grad=False)
        self.v_rest = nn.Parameter(v_rest, requires_grad=False)
        self.tau_m = nn.Parameter(tau_m, requires_grad=False)
        self.tau_g = nn.Parameter(tau_g, requires_grad=False)
        self.pre_activation_coefficient = nn.Parameter(pre_activation_coefficient, requires_grad=False)
        self.post_activation_coefficient = nn.Parameter(post_activation_coefficient, requires_grad=False)

        # init. state variables
        self.v = torch.zeros((self.N,))
        self.g = torch.zeros_like(self.v)  # syn. conductance
        self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step

        # self.to(self.device)

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.g = self.g.clone().detach()
        self.spiked = self.spiked.clone().detach()

    def forward(self, x_in):
        # constant chosen so as to enable spiking for this model
        # I = w g + w x = w(g + x)
        pre_act_in = self.pre_activation_coefficient * torch.add(self.w.matmul(self.g), x_in)
        I = self.post_activation_coefficient * torch.sigmoid(pre_act_in)
        # dv = (v_rest - v + I) / tau_m
        dv = torch.div(torch.add(torch.sub(self.v_rest, self.v), I), self.tau_m)
        self.v = torch.add(self.v, dv)

        self.spiked = torch.round(torch.sigmoid(torch.sub(self.v, self.spike_threshold)))

        spiked = (self.v >= self.spike_threshold).float()  # thresholding when spiked isn't use for grad.s (non-differentiable)
        not_spiked = (spiked - 1.) / -1.  # flips the boolean mat.

        # v = spiked * v_rest + not_spiked * v
        self.v = torch.add(spiked * self.v_rest, not_spiked * self.v)
        dg = -torch.div(self.g, self.tau_g)
        self.g = torch.add(spiked * torch.ones_like(self.g),
                           not_spiked * torch.add(self.g, dg))

        return self.v, self.spiked

