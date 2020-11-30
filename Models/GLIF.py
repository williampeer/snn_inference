import torch
import torch.nn as nn
from torch import FloatTensor as FT

from TORCH_CUSTOM import static_clamp_for


class GLIF(nn.Module):
    parameter_names = ['w', 'E_L', 'tau_m', 'G', 'R_I', 'f_v', 'f_I', 'delta_theta_s', 'b_s', 'a_v', 'b_v', 'theta_inf', 'delta_V', 'I_A']
    parameter_init_intervals = {'E_L': [-62., -46.], 'tau_m': [1.2, 2.5], 'G': [0.7, 0.9], 'R_I': [60., 70.],
                                'f_v': [0.25, 0.35], 'f_I': [0.2, 0.6], 'delta_theta_s': [10., 12.], 'b_s': [0.35, 0.45],
                                'a_v': [0.45, 0.55], 'b_v': [0.45, 0.55], 'theta_inf': [-16., -20.], 'delta_V': [10., 12.],
                                'I_A': [1.2, 2.]}

    def __init__(self, parameters, N=12, w_mean=0.2, w_var=0.15,
                 neuron_types=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1])):
        # use_cuda = torch.cuda.is_available()
        # device = torch.device("cuda" if use_cuda else "cpu")

        super(GLIF, self).__init__()

        if parameters is not None:
            for key in parameters.keys():
                if key == 'tau_m':
                    tau_m = FT(torch.ones((N,)) * parameters[key])
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

        # __constants__ = ['N', 'E_L', 'delta_theta_s', 'b_s', 'a_v', 'b_v', 'theta_inf']
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
        self.tau_m = nn.Parameter(FT(tau_m).clamp(1.2, 3.), requires_grad=True)
        self.G = nn.Parameter(FT(G).clamp(0.1, 0.9), requires_grad=True)
        self.R_I = nn.Parameter(FT(R_I).clamp(40., 55.), requires_grad=True)
        self.f_v = nn.Parameter(FT(f_v).clamp(0.01, 0.99), requires_grad=True)
        self.f_I = nn.Parameter(FT(f_I).clamp(0.01, 0.99), requires_grad=True)
        self.delta_theta_s = nn.Parameter(FT(delta_theta_s).clamp(6., 30.), requires_grad=True)
        self.b_s = nn.Parameter(FT(b_s).clamp(0.01, 0.9), requires_grad=True)
        self.a_v = nn.Parameter(FT(a_v).clamp(0.01, 0.9), requires_grad=True)
        self.b_v = nn.Parameter(FT(b_v).clamp(0.01, 0.9), requires_grad=True)
        self.theta_inf = nn.Parameter(FT(theta_inf).clamp(-25., 0), requires_grad=True)
        self.delta_V = nn.Parameter(FT(delta_V).clamp(0.01, 35.), requires_grad=True)
        self.I_A = nn.Parameter(FT(I_A).clamp(0.5, 3.), requires_grad=True)

        l, m = self.calc_dynamic_clamp_R_I()
        R_I = FT(R_I)
        for i in range(R_I.shape[0]):
            R_I[i].clamp_(float(l[i]), float(m[i]))
        self.R_I = nn.Parameter(R_I, requires_grad=True)

        self.register_backward_clamp_hooks()

        # self.to(self.device)

    def calc_dynamic_clamp_R_I(self):
        I = self.I_additive.matmul(self.self_recurrence_mask * self.w)
        l = torch.ones_like(self.v) * 40.
        m = (self.theta_s + self.theta_v - self.E_L) / I.clamp(min=1e-02)
        return l, m

    def register_backward_clamp_hooks(self):
        def hook_dynamic_R_I_clamp(grad):
            l, m = self.calc_dynamic_clamp_R_I()
            clamped_grad = grad.detach().clone()
            for i in range(grad.shape[0]):
                clamped_grad[i].clamp_(float(l[i] - self.R_I[i]), float(m[i] - self.R_I[i]))
            return clamped_grad

        self.R_I.register_hook(hook_dynamic_R_I_clamp)

        # --------------------------------------
        self.E_L.register_hook(lambda grad: static_clamp_for(grad, -75., -40., self.E_L))
        self.tau_m.register_hook(lambda grad: static_clamp_for(grad, 1.2, 3., self.tau_m))
        self.G.register_hook(lambda grad: static_clamp_for(grad, 0.1, 0.9, self.G))
        self.f_v.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.99, self.f_v))
        self.f_I.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.99, self.f_I))
        self.delta_theta_s.register_hook(lambda grad: static_clamp_for(grad, 6., 30., self.delta_theta_s))
        self.b_s.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.9, self.b_s))
        self.a_v.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.9, self.a_v))
        self.b_v.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.9, self.b_v))
        self.theta_inf.register_hook(lambda grad: static_clamp_for(grad, -25., 0., self.theta_inf))
        self.delta_V.register_hook(lambda grad: static_clamp_for(grad, 0.01, 35., self.delta_V))
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
            # print('DEBUG: p: {}, p.grad: {}'.format(p, p.grad))
        self.reset_hidden_state()

        # self.v = self.E_L.clone().detach() * torch.ones((self.N,))
        # self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step
        # self.theta_s = self.delta_theta_s.clone().detach() * torch.ones((self.N,))
        # self.theta_v = torch.ones((self.N,))
        # self.I_additive = torch.zeros((self.N,))

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.spiked = self.spiked.clone().detach()
        self.theta_s = self.theta_s.clone().detach()
        self.theta_v = self.theta_v.clone().detach()
        self.I_additive = self.I_additive.clone().detach()

    def forward(self, x_in):
        I = self.I_additive.matmul(self.self_recurrence_mask * self.w) + 0.85 * x_in
        # I_A in [0, N * I_A/f_I]
        # sigm(I_A / I_A_max)

        # I = torch.sigmoid(x_in + self.w.matmul(self.I_additive))
        # I = torch.relu(x_in + self.w.matmul(self.I_additive))
        # I = torch.sigmoid((self.self_recurrence_mask * self.w).matmul(self.I_additive) + x_in)

        dv = (I * self.R_I - self.G * (self.v - self.E_L)) / self.tau_m
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

        # return self.v, self.spiked
        return self.spiked
