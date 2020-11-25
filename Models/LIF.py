import torch
import torch.nn as nn
from torch import tensor as T
from torch import FloatTensor as FT


class LIF(nn.Module):
    parameter_names = ['w', 'E_L', 'tau_m', 'R_I', 'tau_g']
    parameter_init_intervals = {'E_L': [-55., -45.], 'tau_m': [1.3, 2.3], 'R_I': [65., 70.], 'tau_g': [2., 3.5]}

    def __init__(self, parameters, N=12, w_mean=0.3, w_var=0.2, neuron_types=T([1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1])):
        super(LIF, self).__init__()
        # self.device = device
        assert len(neuron_types) == N, "neuron_types should be of length N"

        if parameters:
            for key in parameters.keys():
                if key == 'tau_m':
                    tau_m = FT(torch.ones((N,)) * parameters[key])
                elif key == 'E_L':
                    E_L = FT(torch.ones((N,)) * parameters[key])
                elif key == 'tau_g':
                    tau_g = FT(torch.ones((N,)) * parameters[key])
                elif key == 'R_I':
                    R_I = FT(torch.ones((N,)) * parameters[key])
                elif key == 'w_mean':
                    w_mean = float(parameters[key])
                elif key == 'w_var':
                    w_var = float(parameters[key])

        __constants__ = ['spike_threshold', 'N', 'self_recurrence_mask']
        self.spike_threshold = T(30.)
        self.N = N

        self.v = torch.zeros((self.N,))
        self.g = torch.zeros_like(self.v)  # syn. conductance
        self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step

        self.self_recurrence_mask = torch.ones((self.N, self.N)) - torch.eye(self.N, self.N)
        if parameters.__contains__('preset_weights'):
            # print('DEBUG: Setting w to preset weights: {}'.format(parameters['preset_weights']))
            # print('Setting w to preset weights.')
            rand_ws = parameters['preset_weights']
            assert rand_ws.shape[0] == N and rand_ws.shape[1] == N, "shape of weights matrix should be NxN"
        else:
            rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
        for i in range(len(neuron_types)):
            if neuron_types[i] == -1:
                rand_ws[i, :] = -torch.abs(FT(rand_ws[i, :]))
            elif neuron_types[i] == 1:
                rand_ws[i, :] = torch.abs(FT(rand_ws[i, :]))
            else:
                raise NotImplementedError()
        self.w = nn.Parameter(FT(rand_ws), requires_grad=True)  # initialise with positive weights only
        # self.E_L = nn.Parameter(T(N * [E_L]), requires_grad=True)
        # self.tau_m = nn.Parameter(T(N * [tau_m]), requires_grad=True)
        # self.tau_g = nn.Parameter(T(N * [tau_g]), requires_grad=True)
        # self.R_I = nn.Parameter(T(N * [R_I]), requires_grad=True)
        self.E_L = nn.Parameter(FT(E_L), requires_grad=True)  # change to const. if not req. grad to avoid nn.Param parsing
        self.tau_m = nn.Parameter(FT(tau_m), requires_grad=True)
        self.tau_g = nn.Parameter(FT(tau_g), requires_grad=True)
        self.R_I = nn.Parameter(FT(R_I), requires_grad=True)

        for i in range(self.R_I.shape[0]):
            self.R_I[i].register_hook(lambda grad: grad.clamp(float(self.R_I[i]-70 + grad), float(self.R_I[i]-40 + grad)))

        # self.to(self.device)

    def reset(self):
        for p in self.parameters():
            p.grad = None
            # print('DEBUG: p: {}, p.grad: {}'.format(p, p.grad))
        self.reset_hidden_state()

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.g = self.g.clone().detach()
        self.spiked = self.spiked.clone().detach()

    def clamp_parameters(self):
        self.E_L.clamp_(-80., -35.)
        self.tau_m.clamp_(1.15, 3.)
        self.tau_g.clamp_(1.5, 3.5)
        self.R_I.clamp_(40., 70.)
        self.w.clamp_(-1., 1.)
        # row per neuron
        for i in range(len(self.neuron_types)):
            if self.neuron_types[i] == -1:
                self.w[i, :].clamp_(-1., 0.)
            elif self.neuron_types[i] == 1:
                self.w[i, :].clamp_(0., 1.)
            else:
                raise NotImplementedError()

    def dynamic_clamp_R_I(self):
        I = (self.g).matmul(self.self_recurrence_mask * self.w)
        l = torch.ones_like(self.v) * 40.
        m = (torch.ones_like(self.v) * self.spike_threshold - self.E_L) / I

        # for i in range(self.R_I.shape[0]):
        #     self.R_I[i].clamp_(float(l[i]), float(m[i]))

        # self.R_I.clamp(l, m)
        # self.R_I = torch.max(torch.min(self.R_I, m), l)  # manual .clamp

    def forward(self, x_in):
        I = (self.g).matmul(self.self_recurrence_mask * self.w) + 0.9 * x_in
        # I = torch.relu((self.self_recurrence_mask * self.w).matmul(self.g) + x_in)

        dv = (self.E_L - self.v + I * self.R_I) / self.tau_m  # RI - (v - E_L) / tau_m
        v_next = torch.add(self.v, dv)

        # differentiable soft threshold
        self.spiked = torch.sigmoid(torch.sub(v_next, self.spike_threshold))
        # non-differentiable, hard threshold
        spiked = (v_next >= self.spike_threshold).float()
        not_spiked = (spiked - 1.) / -1.

        # v = spiked * E_L + not_spiked * v
        self.v = torch.add(spiked * self.E_L, not_spiked * v_next)
        dg = -torch.div(self.g, self.tau_g)  # -g/tau_g
        self.g = torch.add(spiked * torch.ones_like(self.g), not_spiked * torch.add(self.g, dg))

        # return self.v, self.spiked
        return self.spiked
