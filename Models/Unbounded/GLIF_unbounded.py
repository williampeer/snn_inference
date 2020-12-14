import torch
import torch.nn as nn
from torch import FloatTensor as FT


class GLIF_unbounded(nn.Module):
    parameter_names = ['w', 'E_L', 'tau_m', 'G', 'R_I', 'f_v', 'f_I', 'delta_theta_s', 'b_s', 'a_v', 'b_v', 'theta_inf', 'delta_V', 'I_A']
    parameter_init_intervals = {'E_L': [-62., -46.], 'tau_m': [1.2, 2.5], 'G': [0.7, 0.9], 'R_I': [50., 60.],
                                'f_v': [0.25, 0.35], 'f_I': [0.2, 0.6], 'delta_theta_s': [10., 12.], 'b_s': [0.35, 0.45],
                                'a_v': [0.45, 0.55], 'b_v': [0.45, 0.55], 'theta_inf': [-16., -20.], 'delta_V': [10., 12.],
                                'I_A': [1.2, 2.]}

    def __init__(self, parameters, N=12, w_mean=0.2, w_var=0.15,
                 neuron_types=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1])):
        super(GLIF_unbounded, self).__init__()

        if parameters is not None:
            for key in parameters.keys():
                if key == 'tau_m':
                    tau_m = FT(parameters[key])
                elif key == 'G':
                    G = FT(parameters[key])
                elif key == 'R_I':
                    R_I = FT(parameters[key])
                elif key == 'E_L':
                    E_L = FT(parameters[key])
                elif key == 'delta_theta_s':
                    delta_theta_s = FT(parameters[key])
                elif key == 'b_s':
                    b_s = FT(parameters[key])
                elif key == 'f_v':
                    f_v = FT(parameters[key])
                elif key == 'delta_V':
                    delta_V = FT(parameters[key])
                elif key == 'f_I':
                    f_I = FT(parameters[key])
                elif key == 'I_A':
                    I_A = FT(parameters[key])
                elif key == 'b_v':
                    b_v = FT(parameters[key])
                elif key == 'a_v':
                    a_v = FT(parameters[key])
                elif key == 'theta_inf':
                    theta_inf = FT(parameters[key])

        __constants__ = ['N', 'self_recurrence_mask']
        self.N = N

        self.v = E_L * torch.ones((self.N,))
        self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step
        self.theta_s = delta_theta_s * torch.ones((self.N,))
        self.theta_v = torch.ones((self.N,))
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
        self.E_L = nn.Parameter(FT(E_L).clamp(-75., -40.), requires_grad=True)
        self.tau_m = nn.Parameter(FT(tau_m).clamp(1.1, 3.), requires_grad=True)
        self.G = nn.Parameter(FT(G).clamp(0.1, 0.95), requires_grad=True)
        self.f_v = nn.Parameter(FT(f_v).clamp(0.01, 0.99), requires_grad=True)
        self.f_I = nn.Parameter(FT(f_I).clamp(0.01, 0.99), requires_grad=True)
        self.delta_theta_s = nn.Parameter(FT(delta_theta_s).clamp(6., 30.), requires_grad=True)
        self.b_s = nn.Parameter(FT(b_s).clamp(0.01, 0.9), requires_grad=True)
        self.a_v = nn.Parameter(FT(a_v).clamp(0.01, 0.9), requires_grad=True)
        self.b_v = nn.Parameter(FT(b_v).clamp(0.01, 0.9), requires_grad=True)
        self.theta_inf = nn.Parameter(FT(theta_inf).clamp(-25., 0), requires_grad=True)
        self.delta_V = nn.Parameter(FT(delta_V).clamp(0.01, 35.), requires_grad=True)
        self.I_A = nn.Parameter(FT(I_A).clamp(0.5, 3.), requires_grad=True)

        self.R_I = nn.Parameter(FT(R_I).clamp(25., 64.), requires_grad=True)


    def reset(self):
        for p in self.parameters():
            p.grad = None
        self.reset_hidden_state()

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.spiked = self.spiked.clone().detach()
        self.theta_s = self.theta_s.clone().detach()
        self.theta_v = self.theta_v.clone().detach()
        self.I_additive = self.I_additive.clone().detach()

    def forward(self, x_in):
        I = self.I_additive.matmul(self.self_recurrence_mask * self.w) + 0.85 * x_in

        dv = (I * self.R_I - self.G * (self.v - self.E_L)) / self.tau_m
        v_next = torch.add(self.v, dv)

        # differentiable
        self.spiked = torch.sigmoid(torch.sub(v_next, (self.theta_s + self.theta_v)))
        # NB: Non-differentiable, not used for gradients
        spiked = (v_next >= (self.theta_s + self.theta_v)).float()
        not_spiked = (spiked - 1.) / -1.

        v_reset = self.E_L + self.f_v * (self.v - self.E_L) - self.delta_V
        self.v = spiked * v_reset + not_spiked * v_next  # spike reset

        self.theta_s = (1. - self.b_s) * self.theta_s + spiked * self.delta_theta_s  # always decay
        d_theta_v = self.a_v * (self.v - self.E_L) - self.b_v * (self.theta_v - self.theta_inf)
        self.theta_v = self.theta_v + not_spiked * d_theta_v

        self.I_additive = (1. - self.f_I) * self.I_additive \
                          + self.spiked * self.I_A

        return self.spiked
