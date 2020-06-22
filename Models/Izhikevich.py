import torch
import torch.nn as nn
from torch import tensor


class Izhikevich(nn.Module):
    def __init__(self, device, parameters, N=3, a=0.02, b=0.25, c=-65., d=8., tau_g=6.5, w_mean=0.2, w_var=0.3):
        super(Izhikevich, self).__init__()
        # self.device = device

        if parameters:
            for key in parameters.keys():
                if key == 'tau_g':
                    tau_g = float(parameters[key])
                elif key == 'N':
                    N = int(parameters[key])
                elif key == 'w_mean':
                    w_mean = float(parameters[key])
                elif key == 'w_var':
                    w_var = float(parameters[key])

                elif key == 'a':
                    a = float(parameters[key])
                elif key == 'b':
                    b = float(parameters[key])
                elif key == 'c':
                    c = float(parameters[key])
                elif key == 'd':
                    d = float(parameters[key])

        __constants__ = ['spike_threshold', 'N']
        self.spike_threshold = tensor(30.)
        self.N = N

        self.v = c * torch.ones((self.N,))
        self.u = d * torch.ones((self.N,))
        self.spiked = torch.zeros((self.N,))
        self.g = torch.zeros((self.N,))

        rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
        # self.non_recurrence_mask = torch.ones_like(rand_ws) - torch.eye(self.N, self.N)
        # ws = torch.mul(self.non_recurrence_mask, rand_ws)
        self.w = nn.Parameter(rand_ws, requires_grad=True)  # initialise with positive weights only

        # self.a = T(N * [a])
        # self.b = T(N * [b])
        self.a = nn.Parameter(tensor(N * [a]), requires_grad=True)
        # self.a = nn.Parameter(tensor(a), requires_grad=True)
        # self.b = nn.Parameter(tensor(b), requires_grad=True)
        self.b = nn.Parameter(tensor(N * [b]), requires_grad=True)
        # self.c = T(N * [c])
        # self.d = T(N * [d])
        self.c = nn.Parameter(tensor(N * [c]), requires_grad=True)
        self.d = nn.Parameter(tensor(N * [d]), requires_grad=True)

        self.tau_g = tensor(tau_g)  # synaptic conductance decay constant
        # self.tau_g = nn.Parameter(tau_g * torch.ones((self.N,)), requires_grad=True)

        self.pre_spike_sensitivity = tensor(4.0)
        # self.pre_spike_sensitivity = nn.Parameter(tensor(4.0), requires_grad=True)

        # self.to(self.device)

    def reset_hidden_state(self):
        self.spiked = self.spiked.clone().detach()
        self.v = self.v.clone().detach()
        self.u = self.u.clone().detach()
        self.g = self.g.clone().detach()

    def forward(self, x_in):
        I = torch.add(self.w.matmul(self.g), x_in)
        # these constants may be set to free params
        # dv = tensor(0.04) * torch.pow(self.v, 2) + tensor(5.) * self.v + tensor(140.) - self.u + I
        dv = torch.add(tensor(0.04) * torch.pow(self.v, 2) + tensor(5.) * self.v + tensor(140.) - self.u, I)
        self.v = self.v + dv
        # dv = tensor(0.04) * torch.pow(self.v, 2) + tensor(5.) * self.v + tensor(140.) - self.u + I
        # self.v = self.v + 0.5 * dv

        self.spiked = torch.sigmoid(self.pre_spike_sensitivity * torch.sub(self.v, self.spike_threshold))
        spiked = (self.v >= self.spike_threshold).float()
        not_spiked = (spiked - 1.) / -1.  # not op.

        dg = - torch.div(self.g, self.tau_g)
        du = torch.mul(torch.abs(self.a), torch.sub(torch.mul(torch.abs(self.b), self.v), self.u))

        self.v = not_spiked * self.v + spiked * self.c
        self.u = not_spiked * (self.u + du) + spiked * self.d
        self.g = not_spiked * (self.g + dg) + spiked * torch.ones((self.N,))

        return self.v, self.spiked


class IzhikevichStable(nn.Module):
    def __init__(self, device, parameters, N=3, a=0.02, b=0.25, c=-65., d=8., tau_g=6.5, w_mean=0.2, w_var=0.3):
        super(IzhikevichStable, self).__init__()
        # self.device = device

        if parameters:
            for key in parameters.keys():
                if key == 'tau_g':
                    tau_g = float(parameters[key])
                elif key == 'N':
                    N = int(parameters[key])
                elif key == 'w_mean':
                    w_mean = float(parameters[key])
                elif key == 'w_var':
                    w_var = float(parameters[key])

                elif key == 'a':
                    a = float(parameters[key])
                elif key == 'b':
                    b = float(parameters[key])
                elif key == 'c':
                    c = float(parameters[key])
                elif key == 'd':
                    d = float(parameters[key])

        __constants__ = ['spike_threshold', 'N']
        self.spike_threshold = tensor(30.)
        self.N = N

        self.v = c * torch.ones((self.N,))
        self.u = d * torch.ones((self.N,))
        self.spiked = torch.zeros((self.N,))
        self.g = torch.zeros((self.N,))

        rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
        # self.non_recurrence_mask = torch.ones_like(rand_ws) - torch.eye(self.N, self.N)
        # ws = torch.mul(self.non_recurrence_mask, rand_ws)
        self.w = nn.Parameter(rand_ws, requires_grad=True)  # initialise with positive weights only

        self.a = tensor(N * [a])
        # self.a = nn.Parameter(tensor(N * [a]), requires_grad=True)
        self.b = tensor(N * [b])
        # self.b = nn.Parameter(tensor(N * [b]), requires_grad=True)
        # self.c = tensor(N * [c])
        self.c = nn.Parameter(tensor(N * [c]), requires_grad=True)
        # self.d = tensor(N * [d])
        self.d = nn.Parameter(tensor(N * [d]), requires_grad=True)

        self.tau_g = tensor(tau_g)  # synaptic conductance decay constant
        # self.tau_g = nn.Parameter(tau_g * torch.ones((self.N,)), requires_grad=True)

        self.pre_spike_sensitivity = tensor(6.0)
        # self.pre_spike_sensitivity = nn.Parameter(tensor(6.0), requires_grad=True)

        # self.to(self.device)

    def reset_hidden_state(self):
        self.spiked = self.spiked.clone().detach()
        self.v = self.v.clone().detach()
        self.u = self.u.clone().detach()
        self.g = self.g.clone().detach()

    def forward(self, x_in):
        I = torch.add(self.w.matmul(self.g), x_in)
        # these constants may be set to free params
        dv = tensor(0.04) * torch.pow(self.v, 2) + tensor(5.) * self.v + tensor(140.) - self.u + I
        self.v = self.v + dv
        # dv = tensor(0.04) * torch.pow(self.v, 2) + tensor(5.) * self.v + tensor(140.) - self.u + I
        # self.v = self.v + 0.5 * dv

        self.spiked = torch.sigmoid(self.pre_spike_sensitivity * torch.sub(self.v, self.spike_threshold))
        spiked = (self.v >= self.spike_threshold).float()
        not_spiked = (spiked - 1.) / -1.  # not op.

        dg = - torch.div(self.g, self.tau_g)
        du = torch.mul(torch.abs(self.a), torch.sub(torch.mul(torch.abs(self.b), self.v), self.u))

        self.v = not_spiked * self.v + spiked * self.c
        self.u = not_spiked * (self.u + du) + spiked * self.d
        self.g = not_spiked * (self.g + dg) + spiked * torch.ones((self.N,))

        return self.v, self.spiked



# for instance weights only as free param.
class IzhikevichWeightsOnly(nn.Module):
    def __init__(self, device, parameters, N=3, a=0.02, b=0.25, c=-65., d=8., tau_g=6.5, w_mean=0.2, w_var=0.3):
        super(IzhikevichWeightsOnly, self).__init__()
        # self.device = device

        if parameters:
            for key in parameters.keys():
                if key == 'tau_g':
                    tau_g = float(parameters[key])
                elif key == 'N':
                    N = int(parameters[key])
                elif key == 'w_mean':
                    w_mean = float(parameters[key])
                elif key == 'w_var':
                    w_var = float(parameters[key])

                elif key == 'a':
                    a = float(parameters[key])
                elif key == 'b':
                    b = float(parameters[key])
                elif key == 'c':
                    c = float(parameters[key])
                elif key == 'd':
                    d = float(parameters[key])

        __constants__ = ['spike_threshold', 'N']
        self.spike_threshold = tensor(30.)
        self.N = N

        self.v = c * torch.ones((self.N,))
        self.u = d * torch.ones((self.N,))
        self.spiked = torch.zeros((self.N,))
        self.g = torch.zeros((self.N,))

        rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
        # self.non_recurrence_mask = torch.ones_like(rand_ws) - torch.eye(self.N, self.N)
        # ws = torch.mul(self.non_recurrence_mask, rand_ws)
        self.w = nn.Parameter(rand_ws, requires_grad=True)  # initialise with positive weights only

        self.a = tensor(N * [a])
        self.b = tensor(N * [b])
        # self.a = nn.Parameter(T(N * [a]), requires_grad=True)
        # self.b = nn.Parameter(T(N * [b]), requires_grad=True)
        self.c = tensor(N * [c])
        self.d = tensor(N * [d])
        # self.c = nn.Parameter(T(N * [c]), requires_grad=True)
        # self.d = nn.Parameter(T(N * [d]), requires_grad=True)

        self.tau_g = tensor(tau_g)  # synaptic conductance decay constant
        # self.tau_g = nn.Parameter(tau_g * torch.ones((self.N,)), requires_grad=True)

        self.pre_spike_sensitivity = tensor(6.0)
        # self.pre_spike_sensitivity = nn.Parameter(tensor(4.0), requires_grad=True)

        # self.to(self.device)

    def reset_hidden_state(self):
        self.spiked = self.spiked.clone().detach()
        self.v = self.v.clone().detach()
        self.u = self.u.clone().detach()
        self.g = self.g.clone().detach()

    def forward(self, x_in):
        I = torch.add(self.w.matmul(self.g), x_in)
        # these constants may be set to free params
        dv = tensor(0.04) * torch.pow(self.v, 2) + tensor(5.) * self.v + tensor(140.) - self.u + I
        self.v = self.v + dv
        # dv = tensor(0.04) * torch.pow(self.v, 2) + tensor(5.) * self.v + tensor(140.) - self.u + I
        # self.v = self.v + 0.5 * dv

        self.spiked = torch.sigmoid(self.pre_spike_sensitivity * torch.sub(self.v, self.spike_threshold))
        spiked = (self.v >= self.spike_threshold).float()
        not_spiked = (spiked - 1.) / -1.  # not op.

        dg = - torch.div(self.g, self.tau_g)
        du = torch.mul(torch.abs(self.a), torch.sub(torch.mul(torch.abs(self.b), self.v), self.u))

        self.v = not_spiked * self.v + spiked * self.c
        self.u = not_spiked * (self.u + du) + spiked * self.d
        self.g = not_spiked * (self.g + dg) + spiked * torch.ones((self.N,))

        return self.v, self.spiked
