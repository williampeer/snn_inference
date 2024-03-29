import torch
import torch.nn as nn
from torch import FloatTensor as FT
from torch import tensor as T

from Models.TORCH_CUSTOM import static_clamp_for


class Izhikevich(nn.Module):
    parameter_names = ['w', 'a', 'b', 'c', 'd', 'R_I', 'tau_g']
    parameter_init_intervals = {'a': [0.02, 0.05], 'b': [0.25, 0.27], 'c': [-65., -55.], 'd': [4., 8.], 'R_I': [40., 50.],
                                'tau_g': [2., 3.5]}

    def __init__(self, parameters, N=12, w_mean=0.3, w_var=0.2, neuron_types=T([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1])):
        super(Izhikevich, self).__init__()
        # self.device = device

        if parameters:
            for key in parameters.keys():
                if key == 'tau_g':
                    tau_g = FT(torch.ones((N,)) * parameters[key])

                elif key == 'N':
                    N = int(parameters[key])
                elif key == 'w_mean':
                    w_mean = float(parameters[key])
                elif key == 'w_var':
                    w_var = float(parameters[key])

                elif key == 'a':
                    a = FT(torch.ones((N,)) * parameters[key])
                elif key == 'b':
                    b = FT(torch.ones((N,)) * parameters[key])
                elif key == 'c':
                    c = FT(torch.ones((N,)) * parameters[key])
                elif key == 'd':
                    d = FT(torch.ones((N,)) * parameters[key])
                elif key == 'R_I':
                    R_I = FT(torch.ones((N,)) * parameters[key])

        __constants__ = ['spike_threshold', 'N']
        self.spike_threshold = T(30.)
        self.N = N

        self.v = c * torch.ones((self.N,))
        self.u = d * torch.ones((self.N,))
        self.spiked = torch.zeros((self.N,))
        self.g = torch.zeros((self.N,))

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

        self.a = nn.Parameter(FT(a), requires_grad=True)
        self.b = nn.Parameter(FT(b), requires_grad=True)
        self.c = nn.Parameter(FT(c), requires_grad=True)
        self.d = nn.Parameter(FT(d), requires_grad=True)
        self.tau_g = nn.Parameter(FT(tau_g), requires_grad=True)
        self.R_I = nn.Parameter(FT(R_I), requires_grad=True)

        # self.parameter_names = ['w', 'a', 'b', 'c', 'd', '\\tau_g']
        # self.to(self.device)

    def register_backward_clamp_hooks(self):
        # self.R_I.register_hook(lambda grad: static_clamp_for(grad, 100., 150., self.R_I))
        self.a.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.2, self.E_L))
        self.b.register_hook(lambda grad: static_clamp_for(grad, 0.2, 0.255, self.tau_m))
        self.c.register_hook(lambda grad: static_clamp_for(grad, -80., -50., self.tau_m))
        self.d.register_hook(lambda grad: static_clamp_for(grad, 2., 8., self.tau_m))
        self.tau_g.register_hook(lambda grad: static_clamp_for(grad, 1.15, 3.5, self.tau_g))

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

        self.v = self.c.clone().detach() * torch.ones((self.N,))
        self.v = self.d.clone().detach() * torch.ones((self.N,))
        self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step
        self.g = torch.zeros_like(self.v)  # syn. conductance

    def reset_hidden_state(self):
        self.spiked = self.spiked.clone().detach()
        self.v = self.v.clone().detach()
        self.u = self.u.clone().detach()
        self.g = self.g.clone().detach()

    def forward(self, x_in):
        # I = torch.add(self.w.matmul(self.g), x_in)
        I = (self.g).matmul(self.self_recurrence_mask * self.w) + 0.9 * x_in

        dv = T(0.04) * torch.pow(self.v, 2) + T(5.) * self.v + T(140.) - self.u + I * self.R_I
        v_next = self.v + dv

        self.spiked = torch.sigmoid(torch.sub(v_next, self.spike_threshold))
        spiked = (v_next >= self.spike_threshold).float()
        not_spiked = (spiked - 1.) / -1.  # not op.

        dg = - torch.div(self.g, self.tau_g)
        du = torch.mul(torch.abs(self.a), torch.sub(torch.mul(torch.abs(self.b), self.v), self.u))

        self.v = not_spiked * v_next + spiked * self.c
        self.u = not_spiked * (self.u + du) + spiked * self.d
        self.g = not_spiked * (self.g + dg) + spiked * torch.ones((self.N,))

        # return self.v, self.spiked
        return self.spiked
