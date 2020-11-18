import torch
import torch.nn as nn
from torch import tensor as T
from torch import FloatTensor as FT


class LIF(nn.Module):
    parameter_names = ['w', 'E_L', 'C_m', 'R_I', 'tau_g']
    parameter_init_intervals = {'E_L': [-70., -55.], 'C_m': [1.5, 2.5], 'R_I': [125., 125.], 'tau_g': [1.5, 3.5]}

    def __init__(self, parameters, N=12, w_mean=0.5, w_var=0.5, neuron_types=T([1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1])):
        super(LIF, self).__init__()
        # self.device = device
        assert len(neuron_types) == N, "neuron_types should be of length N"

        if parameters:
            for key in parameters.keys():
                if key == 'C_m':
                    C_m = FT(torch.ones((N,)) * parameters[key])
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
        # self.C_m = nn.Parameter(T(N * [C_m]), requires_grad=True)
        # self.tau_g = nn.Parameter(T(N * [tau_g]), requires_grad=True)
        # self.R_I = nn.Parameter(T(N * [R_I]), requires_grad=True)
        self.E_L = nn.Parameter(FT(E_L), requires_grad=True)  # change to const. if not req. grad to avoid nn.Param parsing
        self.C_m = nn.Parameter(FT(C_m), requires_grad=True)
        self.tau_g = nn.Parameter(FT(tau_g), requires_grad=True)
        self.R_I = nn.Parameter(FT(R_I), requires_grad=True)
        self.E_L.clamp(-80., -35.)
        self.C_m.clamp(1.15, 2.)
        self.tau_g.clamp(2.5, 3.5)
        self.R_I.clamp(90., 150.)
        self.w.clamp(-1., 1.)
        # row per neuron
        for i in range(len(neuron_types)):
            if neuron_types[i] == -1:
                self.w[i, :].clamp(-1., 0.)
            elif neuron_types[i] == 1:
                self.w[i, :].clamp(0., 1.)
            else:
                raise NotImplementedError()

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

    def forward(self, x_in):
        I = torch.sigmoid(
            (self.self_recurrence_mask * self.w).matmul(self.g) + x_in
        )

        dv = (self.E_L - self.v + I * self.R_I) / self.C_m
        v_next = torch.add(self.v, dv)

        # differentiable soft threshold
        self.spiked = torch.sigmoid(torch.sub(v_next, self.spike_threshold))
        # non-differentiable, hard threshold
        spiked = (v_next >= self.spike_threshold).float()
        not_spiked = (spiked - 1.) / -1.

        # v = spiked * E_L + not_spiked * v
        self.v = torch.add(spiked * self.E_L, not_spiked * v_next)
        dg = -torch.div(self.g, self.tau_g)
        self.g = torch.add(spiked * torch.ones_like(self.g), not_spiked * torch.add(self.g, dg))

        # return self.v, self.spiked
        return self.spiked
