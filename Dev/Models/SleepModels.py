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


class IzhikevichStable(nn.Module):
    def __init__(self):
        super(IzhikevichStable, self).__init__()

        __constants__ = ['spike_threshold', 'N']
        self.spike_threshold = T(30.)
        N = 12
        self.N = N

        # w_Wake = torch.ones((4, 1)) * torch.cat([T(4 * [0.]), T(4 * [-.5]), T(4 * [-.25])])
        # w_REM = torch.ones((4, 1)) * torch.cat([T(4 * [0.125]), T(4 * [0.6]), T(4 * [0.])])
        # w_NREM = torch.ones((4, 1)) * torch.cat([T(4 * [-.21]), T(4 * [-.2125]), T(4 * [0.])])
        w_Wake = torch.ones((4, 1)) * torch.cat([T(4 * [0.]), T(4 * [-.4]), T(4 * [-.2])])
        w_REM = torch.ones((4, 1)) * torch.cat([T(4 * [.1]), T(4*[.16]), T(4 * [0.])])
        w_NREM = torch.ones((4, 1)) * torch.cat([T(4 * [-.168]), T(4 * [-.13]), T(4 * [0.])])
        w = torch.cat([w_Wake, w_REM, w_NREM])
        self.w = nn.Parameter(w, requires_grad=False)

        a = torch.cat([T(4*[0.15]), T(4*[0.12]), T(4*[0.1])])
        self.a = nn.Parameter(a, requires_grad=False)
        b = torch.cat([T(4*[0.25]), T(4*[0.245]), T(4*[0.25])])
        self.b = nn.Parameter(b, requires_grad=False)
        c = torch.cat([T(4*[-65.]), T(4*[-42.]), T(4*[-62.])])
        self.c = nn.Parameter(c, requires_grad=False)
        d = torch.cat([T(4*[2.]), T(4*[1.3]), T(4*[2.])])
        self.d = nn.Parameter(d, requires_grad=False)

        tau_g = torch.cat([T(4*[5.]), T(4*[1.]), T(4*[3.0])])  # synaptic conductance decay constant
        # tau_g = torch.cat([T(4*[1.]), T(4*[1.0]), T(4*[1.0])])  # synaptic conductance decay constant
        self.tau_g = nn.Parameter(tau_g, requires_grad=False)

        # pre_spike_sensitivity = torch.cat([T(4 * [6.0]), T(4 * [2.0]), T(4 * [3.])])
        pre_spike_sensitivity = torch.cat([T(4*[6.0]), T(4*[6.0]), T(4*[6.0])])
        self.pre_spike_sensitivity = nn.Parameter(pre_spike_sensitivity, requires_grad=False)

        self.v = c * torch.ones((self.N,))
        self.u = d * torch.ones((self.N,))
        self.spiked = torch.zeros((self.N,))
        self.g = torch.zeros((self.N,))

    def reset_hidden_state(self):
        self.spiked = self.spiked.clone().detach()
        self.v = self.v.clone().detach()
        self.u = self.u.clone().detach()
        self.g = self.g.clone().detach()

    def forward(self, x_in):
        I = torch.add(self.w.matmul(self.g), x_in)
        # these constants may be set to free params
        dv = T(0.04) * torch.pow(self.v, 2) + T(5.) * self.v + T(140.) - self.u + I
        # self.v = self.v + dv
        self.v = self.v + 0.5 * dv
        dv = T(0.04) * torch.pow(self.v, 2) + T(5.) * self.v + T(140.) - self.u + I
        self.v = self.v + 0.5 * dv
        # dv = tensor(0.04) * torch.pow(self.v, 2) + tensor(5.) * self.v + tensor(140.) - self.u + I
        # self.v = self.v + 0.5 * dv

        spiked = (self.v >= self.spike_threshold).float()
        not_spiked = (spiked - 1.) / -1.  # not op.
        # self.spiked = torch.sigmoid(self.pre_spike_sensitivity * torch.sub(self.v, self.spike_threshold))
        self.spiked = spiked

        dg = - torch.div(self.g, self.tau_g)
        du = torch.mul(torch.abs(self.a), torch.sub(torch.mul(torch.abs(self.b), self.v), self.u))

        self.v = not_spiked * self.v + spiked * self.c
        self.u = not_spiked * (self.u + du) + spiked * self.d
        self.g = not_spiked * (self.g + dg) + spiked * torch.ones((self.N,))

        return self.v, self.spiked
