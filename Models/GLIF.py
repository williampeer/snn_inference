import torch
import torch.nn as nn
from torch import FloatTensor as FT


class GLIF(nn.Module):
    parameter_names = ['w', 'E_L', 'C_m', 'G', 'R_I', 'f_v', 'f_I', 'delta_theta_s', 'b_s', 'a_v', 'b_v', 'theta_inf', 'delta_V', 'I_A']
    parameter_init_intervals = {'E_L': [-60., -37.], 'C_m': [1.2, 1.8], 'G': [0.2, 0.9], 'R_I': [145., 150.],
                                'f_v': [0.2, 0.4], 'f_I': [0.1, 0.4], 'delta_theta_s': [5., 30.], 'b_s': [0.2, 0.4],
                                'a_v': [0.2, 0.5], 'b_v': [0.1, 0.5], 'theta_inf': [-20., -10.], 'delta_V': [6., 16.],
                                'I_A': [0.5, 2.]}

    def __init__(self, parameters, N=12, w_mean=0.2, w_var=0.3):
        # use_cuda = torch.cuda.is_available()
        # device = torch.device("cuda" if use_cuda else "cpu")

        super(GLIF, self).__init__()

        if parameters:
            for key in parameters.keys():
                if key == 'C_m':
                    C_m = FT(torch.ones((N,)) * parameters[key])
                elif key == 'G':
                    G = FT(torch.ones((N,)) * parameters[key])
                elif key == 'R_I':
                    R_I = FT(torch.ones((N,)) * parameters[key])
                elif key == 'E_L':
                    E_L = FT(torch.ones((N,)) * parameters[key])
                elif key == 'delta_theta_s':
                    delta_theta_s = FT(torch.ones((N,)) * parameters[key])
                elif key == 'b_s':
                    b_s = FT(torch.ones((N,)) * parameters[key])
                elif key == 'f_v':
                    f_v = FT(torch.ones((N,)) * parameters[key])
                elif key == 'delta_V':
                    delta_V = FT(torch.ones((N,)) * parameters[key])
                elif key == 'f_I':
                    f_I = FT(torch.ones((N,)) * parameters[key])
                elif key == 'I_A':
                    I_A = FT(torch.ones((N,)) * parameters[key])
                elif key == 'b_v':
                    b_v = FT(torch.ones((N,)) * parameters[key])
                elif key == 'a_v':
                    a_v = FT(torch.ones((N,)) * parameters[key])
                elif key == 'theta_inf':
                    theta_inf = FT(torch.ones((N,)) * parameters[key])
                elif key == 'w_mean':
                    w_mean = FT(torch.ones((N,)) * parameters[key])
                elif key == 'w_var':
                    w_var = FT(torch.ones((N,)) * parameters[key])

        __constants__ = ['N', 'E_L', 'delta_theta_s', 'b_s', 'a_v', 'b_v', 'theta_inf']
        self.N = N
        # self.E_L = FT(N * [E_L])

        # self.delta_theta_s = FT(delta_theta_s)
        # self.b_s = FT(b_s)
        # self.a_v = FT(a_v)
        # self.b_v = FT(b_v)
        # self.theta_inf = FT(theta_inf)

        self.v = E_L * torch.ones((self.N,))
        self.g = torch.zeros_like(self.v)  # syn. conductance
        self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step
        self.theta_s = 30. * torch.ones((self.N,))
        self.theta_v = torch.ones((self.N,))
        self.I_additive = torch.zeros((self.N,))

        if parameters.__contains__('preset_weights'):
            # print('DEBUG: Setting w to preset weights: {}'.format(parameters['preset_weights']))
            # print('Setting w to preset weights.')
            rand_ws = parameters['preset_weights']
            assert rand_ws.shape[0] == N and rand_ws.shape[1] == N, "shape of weights matrix should be NxN"
        else:
            rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
        self.w = nn.Parameter(FT(rand_ws), requires_grad=True)
        self.E_L = nn.Parameter(FT(E_L), requires_grad=True)
        self.C_m = nn.Parameter(FT(C_m), requires_grad=True)
        self.G = nn.Parameter(FT(G), requires_grad=True)
        self.R_I = nn.Parameter(FT(R_I), requires_grad=True)
        self.f_v = nn.Parameter(FT(f_v), requires_grad=True)
        self.f_I = nn.Parameter(FT(f_I), requires_grad=True)
        self.delta_theta_s = nn.Parameter(FT(delta_theta_s), requires_grad=True)
        self.b_s = nn.Parameter(FT(b_s), requires_grad=True)
        self.a_v = nn.Parameter(FT(a_v), requires_grad=True)
        self.b_v = nn.Parameter(FT(b_v), requires_grad=True)
        self.theta_inf = nn.Parameter(FT(theta_inf), requires_grad=True)
        self.delta_V = nn.Parameter(FT(delta_V), requires_grad=True)
        self.I_A = nn.Parameter(FT(I_A), requires_grad=True)
        self.w.clamp(-1., 1.)
        self.E_L.clamp(-80., -35.)
        self.C_m.clamp(1.1, 3.)
        self.G.clamp(0.1, 0.9)
        self.R_I.clamp(100., 180.)
        self.f_v.clamp(0.01, 0.99)
        self.f_I.clamp(0.01, 0.99)
        self.delta_theta_s.clamp(6., 30.)
        self.b_s.clamp(0.01, 0.9)
        self.a_v.clamp(0.01, 0.9)
        self.b_v.clamp(0.01, 0.9)
        self.theta_inf.clamp(-25., 0)
        self.delta_V.clamp(0.01, 35.)
        self.I_A.clamp(0.5, 4.)
        # self.I_A = FT(I_A)
        # self.delta_V = FT(delta_V)



    def reset(self):
        for p in self.parameters():
            p.grad = None
            # print('DEBUG: p: {}, p.grad: {}'.format(p, p.grad))
        self.reset_hidden_state()

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.g = self.g.clone().detach()
        self.spiked = self.spiked.clone().detach()
        self.theta_s = self.theta_s.clone().detach()
        self.theta_v = self.theta_v.clone().detach()
        self.I_additive = self.I_additive.clone().detach()

    def forward(self, x_in):
        # I = (x_in + self.w.matmul(self.I_additive))
        I = torch.sigmoid(x_in + self.w.matmul(self.I_additive))
        # I = torch.relu(x_in + self.w.matmul(self.I_additive))

        dv = (I * self.R_I - self.G * (self.v - self.E_L)) / self.C_m
        v_next = self.v + dv

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

        return self.v, self.spiked
