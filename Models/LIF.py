import torch
import torch.nn as nn
from torch import FloatTensor as FT
from torch import tensor as T

from Models.TORCH_CUSTOM import static_clamp_for, static_clamp_for_matrix


class LIF(nn.Module):
    parameter_names = ['w', 'E_L', 'tau_m', 'tau_g']
    parameter_init_intervals = {'E_L': [-64., -55.], 'tau_m': [3.5, 4.0], 'tau_g': [5., 6.] }

    def __init__(self, parameters, N=12, w_mean=0.4, w_var=0.25, neuron_types=T([1, -1])):
        super(LIF, self).__init__()
        # self.device = device

        if parameters:
            for key in parameters.keys():
                if key == 'tau_m':
                    tau_m = FT(torch.ones((N,)) * parameters[key])
                elif key == 'E_L':
                    E_L = FT(torch.ones((N,)) * parameters[key])
                elif key == 'tau_g':
                    tau_g = FT(torch.ones((N,)) * parameters[key])
                elif key == 'w_mean':
                    w_mean = float(parameters[key])
                elif key == 'w_var':
                    w_var = float(parameters[key])

        __constants__ = ['N', 'norm_R_const', 'self_recurrence_mask', 'V_thresh']
        self.N = N
        self.V_thresh = 30.
        self.norm_R_const = 1.1 * (self.V_thresh - E_L)

        self.v = torch.zeros((self.N,))
        self.g = torch.zeros_like(self.v)  # syn. conductance

        self.self_recurrence_mask = torch.ones((self.N, self.N)) - torch.eye(self.N, self.N)
        if parameters.__contains__('preset_weights'):
            # print('Setting w to preset weights.')
            rand_ws = torch.abs(parameters['preset_weights'])
            assert rand_ws.shape[0] == N and rand_ws.shape[1] == N, "shape of weights matrix should be NxN"
        else:
            rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
        nt = T(neuron_types).float()
        self.neuron_types = torch.transpose((nt * torch.ones((self.N, self.N))), 0, 1)
        self.w = nn.Parameter(FT(rand_ws), requires_grad=True)  # initialise with positive weights only
        self.self_recurrence_mask = torch.ones((self.N, self.N)) - torch.eye(self.N, self.N)

        self.E_L = nn.Parameter(FT(E_L).clamp(-80., -35.), requires_grad=True)
        self.tau_m = nn.Parameter(FT(tau_m).clamp(1.5, 8.), requires_grad=True)
        self.tau_g = nn.Parameter(FT(tau_g).clamp(1., 12,), requires_grad=True)

        self.register_backward_clamp_hooks()

    def register_backward_clamp_hooks(self):
        self.E_L.register_hook(lambda grad: static_clamp_for(grad, -80., -35., self.E_L))
        self.tau_m.register_hook(lambda grad: static_clamp_for(grad, 1.5, 8., self.tau_m))
        self.tau_g.register_hook(lambda grad: static_clamp_for(grad, 1., 12., self.tau_g))

        self.w.register_hook(lambda grad: static_clamp_for_matrix(grad, 0., 1., self.w))

    def get_parameters(self):
        params_list = []

        params_list.append(self.w.data)
        params_list.append(self.E_L.data)
        params_list.append(self.tau_m.data)
        params_list.append(self.tau_g.data)

        return params_list

    def reset(self):
        for p in self.parameters():
            p.grad = None
        self.reset_hidden_state()

        self.v = self.E_L.clone().detach() * torch.ones((self.N,))
        self.g = torch.zeros_like(self.v)  # syn. conductance

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.g = self.g.clone().detach()

    def name(self):
        return self.__class__.__name__

    def forward(self, I_ext):
        W_syn = self.w * self.neuron_types
        I_syn = (self.g).matmul(self.self_recurrence_mask * W_syn)

        dv = torch.div(torch.add(torch.sub(self.E_L, self.v), self.norm_R_const*torch.add(I_syn, I_ext)), self.tau_m)
        v_next = torch.add(self.v, dv)

        # self.g = torch.where(v_next >= self.V_thresh, T(1.), dg)
        # self.v = torch.where(v_next >= self.V_thresh, self.E_L, v_next)
        # non-differentiable, hard threshold for nonlinear reset dynamics
        # spiked = (v_next >= self.V_thresh).float()
        spiked = torch.where(v_next >= self.V_thresh, T(1.), T(0.))
        not_spiked = torch.div(torch.sub(spiked, 1.), -1.)

        self.v = torch.add(torch.mul(spiked, self.E_L), torch.mul(not_spiked, v_next))
        dg = torch.sub(self.g, torch.div(self.g, self.tau_g))
        self.g = torch.add(spiked, torch.mul(not_spiked, dg))

        # if spiked.any():
        #     print('hooyray')
        # differentiable soft threshold
        soft_spiked = torch.sigmoid(torch.sub(v_next, self.V_thresh))
        return soft_spiked
