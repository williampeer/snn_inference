import torch
import torch.nn as nn
from torch import FloatTensor as FT

from Models.TORCH_CUSTOM import static_clamp_for, static_clamp_for_matrix


class microGIF(nn.Module):
    parameter_names = ['w', ]
    parameter_init_intervals = { '' }

    def __init__(self, parameters, N=4, neuron_types=[1, -1]):
        super(microGIF, self).__init__()

        if parameters is not None:
            for key in parameters.keys():
                if key == 'tau_m':
                    tau_m = FT(torch.ones((N,)) * parameters[key])
                elif key == 'tau_s':
                    tau_s = FT(torch.ones((N,)) * parameters[key])
                elif key == 'tau_theta':
                    tau_theta = FT(torch.ones((N,)) * parameters[key])
                elif key == 'J_theta':
                    J_theta = FT(torch.ones((N,)) * parameters[key])
                elif key == 'E_L':
                    E_L = FT(torch.ones((N,)) * parameters[key])

        if parameters.__contains__('preset_weights'):
            # print('DEBUG: Setting w to preset weights: {}'.format(parameters['preset_weights']))
            # print('Setting w to preset weights.')
            rand_ws = parameters['preset_weights']
            assert rand_ws.shape[0] == N and rand_ws.shape[1] == N, "shape of weights matrix should be NxN"
        else:
            rand_ws = (0.5 - 0.25) + 2 * 0.25 * torch.rand((self.N, self.N))
        nt = torch.tensor(neuron_types).float()
        self.neuron_types = torch.transpose((nt * torch.ones((self.N, self.N))), 0, 1)
        self.w = nn.Parameter(FT(rand_ws), requires_grad=True)  # initialise with positive weights only
        self.self_recurrence_mask = torch.ones((self.N, self.N)) - torch.eye(self.N, self.N)

        __constants__ = ['N', 'norm_R_const', 'self_recurrence_mask', 'Theta_max']
        self.N = N

        self.v = E_L * torch.ones((self.N,))
        # self.s = torch.zeros_like(self.v)

        self.E_L = nn.Parameter(FT(E_L), requires_grad=True)  # Rest potential
        self.Delta = 0.  # Transmission delay
        self.R_m = 0.
        self.tau_m = nn.Parameter(FT(tau_m), requires_grad=True)
        self.tau_s = nn.Parameter(FT(tau_s), requires_grad=True)
        self.tau_theta = nn.Parameter(FT(tau_theta), requires_grad=True)  # Adaptation time constant
        self.J_theta = nn.Parameter(FT(J_theta), requires_grad=True)  # Adaptation strength
        self.t_refractory = 0.
        self.theta_inf = 0.
        self.reset_potential = 0.
        self.c = 0.
        self.Delta_v = 0.

        self.register_backward_clamp_hooks()

    def reset(self):
        for p in self.parameters():
            p.grad = None
        self.reset_hidden_state()

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        # self.s = self.s.clone().detach()

    def register_backward_clamp_hooks(self):
        self.E_L.register_hook(lambda grad: static_clamp_for(grad, -80., -35., self.E_L))
        self.tau_m.register_hook(lambda grad: static_clamp_for(grad, 1.5, 8., self.tau_m))
        self.tau_s.register_hook(lambda grad: static_clamp_for(grad, 1., 12., self.tau_s))

        self.w.register_hook(lambda grad: static_clamp_for_matrix(grad, 0., 1., self.w))

    def get_parameters(self):
        params_list = []
        # parameter_names = ['w', 'E_L', 'tau_m', 'G', 'f_v', 'f_I', 'delta_theta_s', 'b_s', 'a_v', 'b_v', 'theta_inf', 'delta_V', 'tau_s']
        params_list.append(self.w.data)
        params_list.append(self.E_L.data)
        params_list.append(self.tau_m.data)
        params_list.append(self.tau_s.data)

        return params_list

    def name(self):
        return self.__class__.__name__

    def forward(self, I_ext):
        # dv/dt = (-v + E_L + R * I )/ tau_m
        I_syn = (self.w * (self.epsilon * self.s)) * self.tau_m / self.R_m
        dv = (-self.v + self.E_L + self.R_m * (I_syn + I_ext)) / self.tau_m
        v_next = self.v + dv

        # differentiable
        spikes_lambda = self.c * torch.exp((v_next - self.theta_v) / self.Delta)
        # non-differentiable
        spiked = spikes_lambda.float()
        not_spiked = (spiked - 1.) / -1.

        self.v = not_spiked * v_next + spiked * self.reset_potential

        self.time_since_spike = self.syn_kernel + 1
        self.epsilon = (0.5 + 0.5 * torch.tanh(self.time_since_spike - self.Delta)) * torch.exp(-(self.time_since_spike - self.Delta) / self.tau_s) / self.tau_s

        adaptation_kernel = (self.J/self.tau_theta) * torch.exp(-self.time_since_spike / self.tau_theta)
        dtheta_v = (self.theta_inf -self.theta_v + self.J*self.s) / self.tau_theta
        self.theta_v = self.theta_v + dtheta_v

        return spikes_lambda
