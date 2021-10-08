import torch
import torch.nn as nn
from torch import FloatTensor as FT

from Models.TORCH_CUSTOM import static_clamp_for, static_clamp_for_matrix


class microGIF(nn.Module):
    parameter_names = ['w', 'E_L', 'tau_m', 'tau_s', 'tau_theta', 'J_theta']
    parameter_init_intervals = { 'E_L': [2., 8.], 'tau_m': [6., 15.], 'tau_s': [2., 8.], 'tau_theta': [950., 1050.],
                                 'J_theta': [0.9, 1.1] }
    param_lin_constraints = [[0., 1.], [-10., 30.], [2., 20.], [1.5, 20.], [800., 1500.], [0.5, 1.5]]

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
                elif key == 'R_m':
                    R_m = FT(torch.ones((N,)) * parameters[key])
                elif key == 'c':
                    c = FT(torch.ones((N,)) * parameters[key])
                elif key == 'pop_sizes':
                    pop_sizes = FT(torch.ones((N,)) * parameters[key])

        __constants__ = ['N', 'self_recurrence_mask', 'R_m']
        self.N = N

        if parameters.__contains__('preset_weights'):
            # print('DEBUG: Setting w to preset weights: {}'.format(parameters['preset_weights']))
            # print('Setting w to preset weights.')
            rand_ws = torch.abs(parameters['preset_weights'])
            assert rand_ws.shape[0] == N and rand_ws.shape[1] == N, "shape of weights matrix should be NxN"
        else:
            rand_ws = (0.5 - 0.25) + 2 * 0.25 * torch.rand((self.N, self.N))
        nt = torch.tensor(neuron_types).float()
        self.neuron_types = torch.transpose((nt * torch.ones((self.N, self.N))), 0, 1)
        self.w = nn.Parameter(FT(rand_ws), requires_grad=True)  # initialise with positive weights only
        self.self_recurrence_mask = torch.ones((self.N, self.N)) - torch.eye(self.N, self.N)
        # self.self_recurrence_mask = torch.ones((self.N, self.N))

        self.v = E_L * torch.ones((self.N,))
        self.spiked = torch.zeros((self.N,))
        # self.g = torch.zeros((self.N,))
        self.time_since_spike = torch.zeros((N,))

        self.E_L = nn.Parameter(FT(E_L), requires_grad=True)  # Rest potential
        self.tau_m = nn.Parameter(FT(tau_m), requires_grad=True)
        self.tau_s = nn.Parameter(FT(tau_s), requires_grad=True)
        self.tau_theta = nn.Parameter(FT(tau_theta), requires_grad=True)  # Adaptation time constant
        self.J_theta = nn.Parameter(FT(J_theta), requires_grad=True)  # Adaptation strength
        self.Delta_delay = 1.  # Transmission delay
        self.Delta_u = 5.  # Sensitivity
        self.theta_inf = 15.
        self.reset_potential = 0.
        self.theta_v = FT(self.theta_inf * torch.ones((N,)))
        self.R_m = FT(R_m)
        self.t_refractory = 2.
        self.c = c
        self.pop_sizes = FT(pop_sizes)

        self.register_backward_clamp_hooks()

    def reset(self):
        for p in self.parameters():
            p.grad = None
        self.reset_hidden_state()

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.spiked = self.spiked.clone().detach()
        self.time_since_spike = self.time_since_spike.clone().detach()
        self.theta_v = self.theta_v.clone().detach()

    def register_backward_clamp_hooks(self):
        self.E_L.register_hook(lambda grad: static_clamp_for(grad, -10., 40., self.E_L))
        self.tau_m.register_hook(lambda grad: static_clamp_for(grad, 5., 20., self.tau_m))
        self.tau_s.register_hook(lambda grad: static_clamp_for(grad, 1.5, 20., self.tau_s))
        self.tau_theta.register_hook(lambda grad: static_clamp_for(grad, 650., 1350, self.tau_theta))
        self.J_theta.register_hook(lambda grad: static_clamp_for(grad, 0.1, 4., self.J_theta))

        self.w.register_hook(lambda grad: static_clamp_for_matrix(grad, 0., 1., self.w))

    def get_parameters(self):
        params_dict = {}

        params_dict['w'] = self.w.data
        params_dict['E_L'] = self.E_L.data
        params_dict['tau_m'] = self.tau_m.data
        params_dict['tau_s'] = self.tau_s.data
        params_dict['tau_theta'] = self.tau_theta.data
        params_dict['J_theta'] = self.J_theta.data


        return params_dict

    def name(self):
        return self.__class__.__name__

    def forward(self, I_ext):
        #   adaptation_kernel can be rewritten to:
        dtheta_v = (self.theta_inf - self.theta_v + self.J_theta * self.spiked) / self.tau_theta
        self.theta_v = self.theta_v + dtheta_v

        epsilon = (0.5 + 0.5 * torch.tanh(self.time_since_spike - self.Delta_delay - self.t_refractory)) * torch.exp(
            -(self.time_since_spike - self.Delta_delay - self.t_refractory) / self.tau_s) / self.tau_s

        W_syn = self.self_recurrence_mask * self.w * self.neuron_types
        # Scale synaptic currents with pop sizes.
        I_syn = (self.tau_m * (epsilon * self.pop_sizes * self.spiked).matmul(W_syn)) / self.R_m
        dv = (self.E_L - self.v + self.R_m * (I_syn + I_ext)) / self.tau_m
        v_next = self.v + dv

        # TODO: Differentiability
        spikes_lambda = self.c * torch.exp((v_next - self.theta_v) / self.Delta_u)
        m = torch.distributions.bernoulli.Bernoulli(spikes_lambda)
        spiked = m.sample()
        # spiked = torch.bernoulli(spikes_lambda)
        # spiked = m.rsample()
        # spiked = torch.sigmoid(100.*(spikes_lambda-draw))
        # spiked = torch.sigmoid(spikes_lambda-0.1)
        self.spiked = spiked
        # self.spiked = spiked
        not_spiked = (spiked - 1.) / -1.

        self.time_since_spike = not_spiked * (self.time_since_spike + 1)
        self.v = not_spiked * v_next + spiked * self.reset_potential

        # return m.log_prob(spiked)
        # return spikes_lambda
        # return spiked
        return spikes_lambda, spiked
        # return self.v, spiked