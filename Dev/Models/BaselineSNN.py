import torch
from torch import nn
from torch import tensor


class BaselineSNN(nn.Module):
    def __init__(self, device='cpu', parameters={}, N=3, w_mean=0.2, w_var=0.3):
        super(BaselineSNN, self).__init__()

        if parameters:
            for key in parameters.keys():
                if key == 'w_mean':
                    w_mean = float(parameters[key])
                elif key == 'w_var':
                    w_var = float(parameters[key])
                elif key == 'N':
                    N = int(parameters[key])

        self.N = N

        self.v = -65. * torch.ones((N,))
        self.spiked = torch.ones((N,))

        rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((N, N))
        self.w = nn.Parameter(rand_ws, requires_grad=True)

    def reset_hidden_state(self):
        self.spiked = self.spiked.clone().detach()
        self.v = self.v.clone().detach()

    def forward(self, x_in):
        v_rest = tensor(-65.); I_coeff = tensor(130.); tau_m = tensor(6.5); spike_threshold = tensor(30.)

        I = torch.sigmoid(x_in + torch.matmul(self.spiked, self.w))
        dv = (v_rest - self.v +  I_coeff*I) / tau_m
        self.v = self.v + dv

        self.spiked = torch.sigmoid(6.0 * torch.sub(self.v, spike_threshold))
        spiked = (self.v >= spike_threshold).float()
        not_spiked = (spiked -1./-1.)
        # not_spiked = (self.spiked -1./-1.)

        self.v = not_spiked * self.v + spiked * v_rest

        return self.v, self.spiked
