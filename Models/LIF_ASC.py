import torch
import torch.nn as nn
from torch import FloatTensor as FT
from torch import tensor as T

from Models.TORCH_CUSTOM import static_clamp_for


class LIF_ASC(nn.Module):
    parameter_names = ['w', 'E_L', 'tau_m', 'tau_s', 'G', 'f_v', 'b_s', 'tau_s']
    parameter_init_intervals = {'E_L': [-66., -45.], 'tau_m': [1.9, 2.6], 'G': [0.7, 0.9], 'f_I': [0.3, 0.5],
                                'f_v': [0.2, 0.4], 'b_s': [0.2, 0.4], 'I_A': [1.2, 1.5], 'tau_s': [3.5, 5.5]}

    def __init__(self, parameters, N=12, w_mean=0.3, w_var=0.2, neuron_types=T([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1])):
        super(LIF_ASC, self).__init__()
        # self.device = device

        if parameters:
            for key in parameters.keys():
                if key == 'tau_m':
                    tau_m = FT(torch.ones((N,)) * parameters[key])
                elif key == 'tau_s':
                    tau_s = FT(torch.ones((N,)) * parameters[key])
                elif key == 'E_L':
                    E_L = FT(torch.ones((N,)) * parameters[key])
                elif key == 'G':
                    G = FT(torch.ones((N,)) * parameters[key])
                elif key == 'N':
                    N = int(parameters[key])
                elif key == 'w_mean':
                    w_mean = float(parameters[key])
                elif key == 'w_var':
                    w_var = float(parameters[key])
                elif key == 'delta_theta_s':
                    delta_theta_s = FT(torch.ones((N,)) * parameters[key])
                elif key == 'b_s':
                    b_s = FT(torch.ones((N,)) * parameters[key])
                elif key == 'f_I':
                    f_I = FT(torch.ones((N,)) * parameters[key])
                elif key == 'I_A':
                    I_A = FT(torch.ones((N,)) * parameters[key])

        __constants__ = ['spike_threshold', 'N', 'norm_R_f', 'self_recurrence_mask']
        self.spike_threshold = T(30.)
        self.N = N

        R_const = 1.1
        self.norm_R_const = (delta_theta_s - E_L) * R_const

        self.v = torch.zeros((self.N,))
        self.s = torch.zeros_like(self.v)  # syn. conductance
        # self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step
        # self.theta_s = delta_theta_s * torch.ones((self.N,))
        self.I_additive = torch.zeros((self.N,))

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
        self.neuron_types = neuron_types
        self.w = nn.Parameter(FT(rand_ws), requires_grad=True)  # initialise with positive weights only

        self.G = nn.Parameter(FT(G).clamp(0.01, 0.99), requires_grad=True)
        self.b_s = nn.Parameter(FT(b_s).clamp(0.01, 0.95), requires_grad=True)
        self.tau_m = nn.Parameter(FT(tau_m).clamp(1.5, 6.), requires_grad=True)
        self.tau_s = nn.Parameter(FT(tau_s).clamp(1., 12.), requires_grad=True)
        self.E_L = nn.Parameter(FT(E_L).clamp(-80., -35.), requires_grad=True)
        self.f_I = nn.Parameter(FT(f_I).clamp(0.01, 0.99), requires_grad=True)

        self.delta_theta_s = nn.Parameter(FT(delta_theta_s).clamp(6., 30.), requires_grad=True)
        self.I_A = nn.Parameter(FT(I_A).clamp(0.5, 3.), requires_grad=True)

    def register_backward_clamp_hooks(self):
        self.E_L.register_hook(lambda grad: static_clamp_for(grad, -75., -40., self.E_L))
        self.tau_m.register_hook(lambda grad: static_clamp_for(grad, 1.1, 3., self.tau_m))
        self.tau_s.register_hook(lambda grad: static_clamp_for(grad, 1., 12., self.tau_m))
        self.G.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.99, self.G))
        self.f_I.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.99, self.f_I))
        self.delta_theta_s.register_hook(lambda grad: static_clamp_for(grad, 6., 30., self.delta_theta_s))
        self.b_s.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.95, self.b_s))
        self.I_A.register_hook(lambda grad: static_clamp_for(grad, 0.5, 3., self.I_A))

        # row per neuron
        for i in range(len(self.neuron_types)):
            if self.neuron_types[i] == -1:
                self.w[i, :].register_hook(lambda grad: static_clamp_for(grad, -1., 0., self.w[i, :]))
            elif self.neuron_types[i] == 1:
                self.w[i, :].register_hook(lambda grad: static_clamp_for(grad, 0., 1., self.w[i, :]))
            else:
                raise NotImplementedError()

    def reset(self):
        for p in self.parameters():
            p.grad = None
        self.reset_hidden_state()

        self.v = self.E_L.clone().detach() * torch.ones((self.N,))
        # self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step
        self.s = torch.zeros_like(self.v)  # spike prop. for next time-step

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        # self.spiked = self.spiked.clone().detach()
        self.s = self.s.clone().detach()
        self.theta_s = self.theta_s.clone().detach()
        self.I_additive = self.I_additive.clone().detach()

    def forward(self, x_in):
        # assuming input weights to be Eye(N,N)
        I = (self.I_additive + self.s).matmul(self.self_recurrence_mask * self.w) + 1.75 * x_in
        dv = (self.G * (self.E_L - self.v) + I * self.norm_R_const) / self.tau_m
        v_next = self.v + dv

        gating = (v_next / self.spike_threshold).clamp(0., 1.)
        dv_max = (self.spike_threshold - self.E_L)
        ds = (-self.s + gating * (dv / dv_max)) / self.tau_s
        self.s = self.s + ds

        # non-differentiable, hard threshold
        spiked = (v_next >= self.spike_threshold).float()
        not_spiked = (spiked - 1.) / -1.

        self.v = torch.add(spiked * self.E_L, not_spiked * v_next)

        # self.I_additive = (1. - self.f_I) * self.I_additive + spiked * self.I_A
        self.I_additive = self.I_additive - self.f_I * self.I_additive + spiked * self.f_I

        return self.v, self.s * self.tau_s
        # return self.s * self.tau_s  # use synaptic current as spike signal

        # return self.v, self.spiked
        # return self.spiked
