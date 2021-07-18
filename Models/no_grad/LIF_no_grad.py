import torch
import torch.nn as nn
from torch import FloatTensor as FT
from torch import tensor as T

from Models.LIF import LIF
from Models.TORCH_CUSTOM import static_clamp_for, static_clamp_for_matrix


class LIF_no_grad(nn.Module):
    parameter_names = ['w', 'E_L', 'tau_m', 'tau_s']
    # parameter_init_intervals = {'E_L': [-65., -52.], 'tau_m': [1.9, 2.3], 'tau_s': [3., 4.3]}
    parameter_init_intervals = {'E_L': [-60., -60.], 'tau_m': [1.6, 1.6], 'tau_s': [2.5, 2.5]}

    def __init__(self, parameters, N=12, w_mean=0.4, w_var=0.25,
                 neuron_types=[1., 1., 1., 1., 1., 1., 1., 1., -1., -1., -1., -1.]):
        super(LIF_no_grad, self).__init__()
        # self.device = device
        assert len(neuron_types) == N, "neuron_types should be of length N"

        if parameters:
            for key in parameters.keys():
                if key == 'tau_m':
                    tau_m = FT(torch.ones((N,)) * parameters[key])
                elif key == 'E_L':
                    E_L = FT(torch.ones((N,)) * parameters[key])
                elif key == 'tau_s':
                    tau_s = FT(torch.ones((N,)) * parameters[key])

        __constants__ = ['spike_threshold', 'N', 'norm_R_const', 'self_recurrence_mask']
        self.spike_threshold = T(30.)
        self.N = N

        R_const = 1.1
        self.norm_R_const = (self.spike_threshold - E_L) * R_const
        # assert not any(tau_m <= 2*self.R_const), "tau_m > 2*R_const for system stability. see forward()"

        self.v = E_L * torch.ones((self.N,))
        self.s = torch.zeros_like(self.v)  # syn. conductance

        self.self_recurrence_mask = torch.ones((self.N, self.N)) - torch.eye(self.N, self.N)
        if parameters.__contains__('preset_weights'):
            # print('DEBUG: Setting w to preset weights: {}'.format(parameters['preset_weights']))
            # print('Setting w to preset weights.')
            rand_ws = torch.abs(parameters['preset_weights'])
            assert rand_ws.shape[0] == N and rand_ws.shape[1] == N, "shape of weights matrix should be NxN. ws.shape: {}".format(rand_ws.shape)
        else:
            rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
            rand_ws = rand_ws.clamp(-1., 1.)
        nt = T(neuron_types).float()
        self.neuron_types = torch.transpose((nt * torch.ones((self.N, self.N))), 0, 1)
        self.w = FT(rand_ws)

        self.E_L = FT(E_L)
        self.tau_m = FT(tau_m)
        self.tau_s = FT(tau_s)

    def reset(self):
        for p in self.parameters():
            p.grad = None
        self.reset_hidden_state()

        self.v = self.E_L.clone().detach() * torch.ones((self.N,))
        self.s = torch.zeros_like(self.v)  # syn. conductance

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.s = self.s.clone().detach()

    def name(self):
        return LIF.__name__

    def get_parameters(self):
        params_list = []
        # parameter_names = ['w', 'E_L', 'tau_m', 'tau_s']
        params_list.append(self.w.data)
        params_list.append(self.E_L.data)
        params_list.append(self.tau_m.data)
        params_list.append(self.tau_s.data)

        return params_list

    # Assuming normalised input.
    def forward(self, x_in):
        W_syn = self.w * self.neuron_types
        I = (self.s).matmul(self.self_recurrence_mask * W_syn) + 1.75 * x_in  # assuming input weights to be Eye(N,N)
        dv = (self.E_L - self.v + I * self.norm_R_const) / self.tau_m
        v_next = torch.add(self.v, dv)

        # gating = (torch.functional.F.relu(v_next) / self.spike_threshold).clamp(0., 1.)
        gating = (v_next / self.spike_threshold).clamp(0., 1.)
        dv_max = (self.spike_threshold - self.E_L)
        ds = (-self.s + gating * (dv / dv_max).clamp(0., 1.)) / self.tau_s
        # ds = (-self.s + torch.functional.F.relu(dv / dv_max)) / self.tau_s
        # ds = (-self.s + gating) / self.tau_s
        self.s = self.s + ds
        v_next = torch.add(self.v, dv)

        # non-differentiable, hard threshold for nonlinear reset dynamics
        spiked = (v_next >= self.spike_threshold).float()
        not_spiked = (spiked - 1.) / -1.

        self.v = torch.add(spiked * self.E_L, not_spiked * v_next)

        soft_spiked = torch.sigmoid(torch.sub(v_next, self.spike_threshold))
        return soft_spiked
        # return gating
        # return self.v, self.s * (self.tau_s)
        # return self.s * self.tau_s  # return readout of synaptic current as spike signal
        # return self.v, self.s * (self.tau_s + 1)/2.  # return readout of synaptic current as spike signal
        # return self.s * (self.tau_s + 1)/2.  # return readout of synaptic current as spike signal
