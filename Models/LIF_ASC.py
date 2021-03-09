import torch
import torch.nn as nn
from torch import tensor as T
from torch import FloatTensor as FT

from Models.TORCH_CUSTOM import static_clamp_for


class LIF_ASC(nn.Module):
    parameter_names = ['w', 'E_L', 'tau_m', 'tau_g', 'G', 'R_I', 'f_v', 'delta_theta_s', 'b_s']
    parameter_init_intervals = {'E_L': [-62., -40.], 'tau_m': [1.2, 2.5], 'G': [0.5, 0.9], 'R_I': [55., 59.],
                                'f_v': [0.2, 0.4], 'delta_theta_s': [10., 20.], 'b_s': [0.2, 0.4]}

    def __init__(self, parameters, N=12, w_mean=0.3, w_var=0.2, neuron_types=T([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1])):
        super(LIF_ASC, self).__init__()
        # self.device = device

        if parameters:
            for key in parameters.keys():
                if key == 'tau_m':
                    tau_m = FT(torch.ones((N,)) * parameters[key])
                elif key == 'E_L':
                    E_L = FT(torch.ones((N,)) * parameters[key])
                elif key == 'G':
                    G = FT(torch.ones((N,)) * parameters[key])
                elif key == 'R_I':
                    R_I = FT(torch.ones((N,)) * parameters[key])
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
                elif key == 'f_v':
                    f_v = FT(torch.ones((N,)) * parameters[key])
                elif key == 'f_I':
                    f_I = FT(torch.ones((N,)) * parameters[key])
                elif key == 'I_A':
                    I_A = FT(torch.ones((N,)) * parameters[key])
                elif key == 'delta_V':
                    delta_V = FT(torch.ones((N,)) * parameters[key])

        __constants__ = ['N']
        self.N = N
        self.G = FT(G)

        self.v = torch.zeros((self.N,))
        self.g = torch.zeros_like(self.v)  # syn. conductance
        self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step
        self.theta_s = delta_theta_s * torch.ones((self.N,))
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

        self.b_s = nn.Parameter(FT(b_s).clamp(0.01, 0.9), requires_grad=True)
        self.tau_m = nn.Parameter(FT(tau_m).clamp(1.1, 3.), requires_grad=True)
        self.E_L = nn.Parameter(FT(E_L).clamp(-80., -35.), requires_grad=True)
        self.f_I = nn.Parameter(FT(f_I).clamp(0.01, 0.99), requires_grad=True)
        self.R_I = nn.Parameter(FT(R_I), requires_grad=True)

        self.delta_theta_s = nn.Parameter(FT(delta_theta_s).clamp(6., 30.), requires_grad=True)
        self.I_A = nn.Parameter(FT(I_A).clamp(0.5, 3.), requires_grad=True)

    def register_backward_clamp_hooks(self):
        self.R_I.register_hook(lambda grad: static_clamp_for(grad, 50., 150., self.R_I))
        self.E_L.register_hook(lambda grad: static_clamp_for(grad, -75., -40., self.E_L))
        self.tau_m.register_hook(lambda grad: static_clamp_for(grad, 1.1, 3., self.tau_m))
        self.G.register_hook(lambda grad: static_clamp_for(grad, 0.1, 0.9, self.G))
        self.f_I.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.99, self.f_I))
        self.delta_theta_s.register_hook(lambda grad: static_clamp_for(grad, 6., 30., self.delta_theta_s))
        self.b_s.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.9, self.b_s))
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
        self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.spiked = self.spiked.clone().detach()
        self.theta_s = self.theta_s.clone().detach()
        self.I_additive = self.I_additive.clone().detach()

    def forward(self, x_in):
        I = self.I_additive.matmul(self.self_recurrence_mask * self.w) + 0.9 * x_in

        dv = (self.G * (self.E_L - self.v) + I * self.R_I) / self.tau_m
        v_next = self.v + dv

        # differentiable
        self.spiked = torch.sigmoid(torch.sub(v_next, self.theta_s))
        # non-differentiable, hard threshold
        spiked = (v_next >= self.theta_s).float()
        not_spiked = (spiked - 1.) / -1.

        self.v = torch.add(spiked * self.E_L, not_spiked * v_next)

        theta_s_next = (1 - self.b_s) * self.theta_s
        self.theta_s = spiked * (self.theta_s + self.delta_theta_s) + not_spiked * theta_s_next

        I_additive_decayed = (torch.ones_like(self.f_I) - self.f_I) * self.I_additive
        self.I_additive = spiked * (self.I_additive + self.I_A) + not_spiked * I_additive_decayed

        # return self.v, self.spiked
        return self.spiked
