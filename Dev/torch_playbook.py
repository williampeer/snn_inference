import torch
import torch.nn as nn
from torch import tensor as T

from eval import calculate_loss
from experiments import poisson_input
from model_util import feed_inputs_sequentially_return_spike_train


class test_LIF(nn.Module):
    def __init__(self, C_m=1.0, tau_g=2.0, E_L=-65., N=4, w_mean=0.15, w_var=0.25, R_I=42.):

        super(test_LIF, self).__init__()

        __constants__ = ['spike_threshold', 'N']
        self.spike_threshold = T(30.)
        self.N = N

        self.v = torch.zeros((self.N,))
        self.g = torch.zeros_like(self.v)
        self.spiked = torch.zeros_like(self.v)

        rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
        self.w = nn.Parameter(rand_ws, requires_grad=True)
        self.E_L = nn.Parameter(T(N * [E_L]), requires_grad=True)
        self.C_m = nn.Parameter(T(N * [C_m]), requires_grad=True)
        self.tau_g = nn.Parameter(T(N * [tau_g]), requires_grad=True)
        self.R_I = nn.Parameter(T(N * [R_I]), requires_grad=True)

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.g = self.g.clone().detach()
        self.spiked = self.spiked.clone().detach()

    def forward(self, x_in):
        I = self.w.matmul(self.g) + x_in

        dv = (self.E_L - self.v + I * self.R_I) / self.C_m
        v_next = torch.add(self.v, dv)

        self.spiked = torch.sigmoid(torch.sub(v_next, self.spike_threshold))
        spiked = (v_next >= self.spike_threshold).float()
        not_spiked = (spiked - 1.) / -1.

        self.v = torch.add(spiked * self.E_L, not_spiked * v_next)
        dg = -torch.div(self.g, self.tau_g)
        self.g = torch.add(spiked * torch.ones_like(self.g), not_spiked * torch.add(self.g, dg))

        return self.v, self.spiked


time_interval = 2000


def custom_fn(input_rate):
    return


# @profile
def train_model():
    rate = torch.tensor(0.6, requires_grad=True)
    sut_grads = []
    model = test_LIF()
    input_variables = []
    model_inputs = custom_fn(input_variables)
    poisson_inputs = poisson_input(rate, t=time_interval, N=4)
    poisson_inputs.retain_grad()
    # perm_ins = p_ins.clone().detach()
    # perm_ins.requires_grad = True

    opt_params = list(model.parameters())
    opt_params.append(rate)
    # opt_params.append(perm_ins)
    optim = torch.optim.Adam(opt_params, lr=0.01)

    for t_i in range(5):
        out_spikes = feed_inputs_sequentially_return_spike_train(model, poisson_inputs)
        # out_spikes = feed_inputs_sequentially_return_spiketrain(model, perm_ins)
        loss = calculate_loss(out_spikes, torch.randint(0, 1, (time_interval, 4), dtype=torch.float), 'van_rossum_dist', tau_vr=torch.tensor(4.0))
        print()

        print('optim.zero_grad()')
        optim.zero_grad()  # what does this do?
        # print_grads(model, rate)
        # print(list(model.parameters())[0])
        print()

        print('loss.backward()')
        loss.backward(retain_graph=True)  # different param options..

        print('model.w.grad.clone().detach().numpy()', model.w.grad.clone().detach().numpy())
        # print(list(model.parameters())[0])
        print('perm_ins grad', poisson_inputs.grad)  # works
        print('perm_ins grad sum', poisson_inputs.grad.sum())  # works
        # rate.grad = poisson_inputs.grad.sum()
        rate.grad = torch.mean(poisson_inputs.grad)
        print('============= rate grad', rate.grad)  # TODO: fix.
        print()

        print('optim.step()')
        optim.step()
        # print_grads(model, rate)
        print('weights parameter:', list(model.parameters())[0])
        print('Poisson rate parameter:', rate)

        model.reset_hidden_state()
        model.load_state_dict(model.state_dict())
        for p_i, param in enumerate(list((model.parameters()))):
            sut_grads.append(param.grad.clone().detach().numpy())
            param.grad = None
        poisson_inputs.grad = None; rate.grad = None
        loss = None
        print('grads set to None')

    sut_states = model.state_dict()
    # loss = None; model = None; optim = None; out_spikes = None; p_ins = None; opt_params = None
    # del loss, model, optim, out_spikes, p_ins, opt_params
    return sut_states, sut_grads


states = []; grads = []
for j in range(5):
    states_i, grads_i = train_model()
    states.append(states_i)
    grads.append(grads_i)
