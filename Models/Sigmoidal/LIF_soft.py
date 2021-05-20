import torch
import torch.nn as nn
from torch import FloatTensor as FT
from torch import tensor as T

from Models.TORCH_CUSTOM import static_clamp_for, static_clamp_for_matrix


class LIF_soft(nn.Module):
    parameter_names = ['w', 'E_L', 'tau_m', 'tau_g']
    # parameter_init_intervals = {'E_L': [-55., -48.], 'tau_m': [1.9, 2.3], 'tau_g': [3., 4.5]}
    # parameter_init_intervals = {'E_L': [-50., -50.], 'tau_m': [2., 2.], 'tau_g': [3.2, 3.2]}
    const_tau = 1.8
    parameter_init_intervals = {'E_L': [-50., -50.], 'tau_m': [const_tau, const_tau], 'tau_g': [const_tau, const_tau]}

    def __init__(self, parameters, N=12, w_mean=0.2, w_var=0.15, neuron_types=T([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1])):
        super(LIF_soft, self).__init__()
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
                # elif key == 'R_I':
                #     R_I = FT(torch.ones((N,)) * parameters[key])
                elif key == 'w_mean':
                    w_mean = float(parameters[key])
                elif key == 'w_var':
                    w_var = float(parameters[key])

        __constants__ = ['spike_threshold', 'N', 'norm_R_const']
        self.spike_threshold = T(30.)
        self.N = N

        R_const = 1.5
        self.norm_R_const = (self.spike_threshold - E_L) * R_const
        # assert not any(tau_m <= 2*self.R_const), "tau_m > 2*R_const for system stability. see forward()"

        self.v = E_L * torch.ones((self.N,))
        self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step
        self.g = torch.zeros_like(self.v)  # syn. conductance

        self.self_recurrence_mask = torch.ones((self.N, self.N)) - torch.eye(self.N, self.N)
        if parameters.__contains__('preset_weights'):
            # print('DEBUG: Setting w to preset weights: {}'.format(parameters['preset_weights']))
            # print('Setting w to preset weights.')
            rand_ws = parameters['preset_weights']
            assert rand_ws.shape[0] == N and rand_ws.shape[1] == N, "shape of weights matrix should be NxN"
        else:
            rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
        nt = T(neuron_types).float()
        self.neuron_types = torch.transpose((nt * torch.ones((self.N, self.N))), 0, 1)
        self.w = nn.Parameter(FT(rand_ws), requires_grad=True)
        # self.w = nn.Parameter(FT(torch.zeros((N,N))), requires_grad=True)

        # self.R_I = nn.Parameter(FT(R_I).clamp(100., 150.), requires_grad=True)  # change to const. if not req. grad to avoid nn.Param parsing
        self.E_L = nn.Parameter(FT(E_L).clamp(-80., -35.), requires_grad=True)  # change to const. if not req. grad to avoid nn.Param parsing
        self.tau_m = nn.Parameter(FT(tau_m).clamp(1.5, 8.), requires_grad=True)
        self.tau_g = nn.Parameter(FT(tau_g).clamp(1.5, 12.), requires_grad=True)
        # self.E_L = FT(E_L).clamp(-80., -35.)  # change to const. if not req. grad to avoid nn.Param parsing
        # self.tau_m = FT(tau_m).clamp(1.5, 8.)
        # self.tau_g = FT(tau_g).clamp(1.5, 12.)

        self.register_backward_clamp_hooks()

    def register_backward_clamp_hooks(self):
        # self.R_I.register_hook(lambda grad: static_clamp_for(grad, 100., 150., self.R_I))
        self.E_L.register_hook(lambda grad: static_clamp_for(grad, -80., -35., self.E_L))
        self.tau_m.register_hook(lambda grad: static_clamp_for(grad, 1.5, 8., self.tau_m))
        self.tau_g.register_hook(lambda grad: static_clamp_for(grad, 1.5, 12., self.tau_g))

        self.w.register_hook(lambda grad: static_clamp_for_matrix(grad, 0., 1., self.w))

    def reset(self):
        for p in self.parameters():
            p.grad = None
        self.reset_hidden_state()

        self.v = self.E_L.clone().detach() * torch.ones((self.N,))
        self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step
        self.g = torch.zeros_like(self.v)  # syn. conductance

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.g = self.g.clone().detach()
        self.spiked = self.spiked.clone().detach()

    # Assuming normalised input.
    def forward(self, x_in):
        W_syn = self.w * self.neuron_types
        I = (self.g).matmul(self.self_recurrence_mask * W_syn) + x_in  # assuming input weights to be Eye(N,N)
        dv = (self.E_L - self.v + I * self.norm_R_const) / self.tau_m
        v_next = torch.add(self.v, dv)

        # differentiable soft threshold
        self.spiked = torch.sigmoid(torch.sub(v_next, self.spike_threshold))
        # non-differentiable, hard threshold
        spiked = (v_next >= self.spike_threshold).float()
        not_spiked = (spiked - 1.) / -1.

        self.v = torch.add(spiked * self.E_L, not_spiked * v_next)
        dg = -torch.div(self.g, self.tau_g)  # -g/tau_g
        self.g = torch.add(spiked * torch.ones_like(self.g), not_spiked * torch.add(self.g, dg))

        return self.spiked
        # return self.v, self.spiked
