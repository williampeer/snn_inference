import torch
from torch import tensor

from Models.BaselineSNN import BaselineSNN
from Models.Izhikevich import Izhikevich, IzhikevichWeightsOnly
from Models.LIF import LIF
from experiments import poisson_input
from model_util import feed_inputs_sequentially_return_spike_train
from plot import plot_all_param_pairs_with_variance
from spike_metrics import van_rossum_dist, euclid_dist


N = 10
gen_model = BaselineSNN(N=N)  # weights is the only free parameter
model = BaselineSNN(N=N)
# model = LIF(device='cpu', parameters={}, N=N)
# model = Izhikevich_constrained(device='cpu', parameters={}, N=N, a=0.1, b=0.29)
# model = Izhikevich(device='cpu', parameters={}, N=N, a=0.1, b=0.29)

optimiser = torch.optim.SGD(list(model.parameters()), lr=0.1)
# optimiser.zero_grad()
# for i in range(100):
#     x_in = 3.*torch.rand((N,))
#     _, spiked = model(x_in)
#     targets = (torch.rand((N,)) > 0.5).float()
#
#     # loss = torch_van_rossum_dist(spikes=spiked, target_spikes=targets, tau=T(5.0))
#     loss = torch.sqrt(torch.pow(torch.sub(spiked, targets), 2).sum() + 1e-18)  # avoid sqrt(0) -> NaN
#     print('loss: {}'.format(loss))
#     if spiked.sum() > 0:
#         print('HOOORAAAY. spike.')
#     # if i % 10 == 0:
#     for param_i, param in enumerate(list(model.parameters())):
#         if param.grad is not None and param.grad.sum() > 0:
#             # print('parameter #{}: {}'.format(param_i, param))
#             print('========== parameter #{} gradient: {}'.format(param_i, param.grad))
#
#     loss.backward(retain_graph=True)
#     optimiser.step()
#     # model.reset_hidden_state()

print('=================== batch sim. ======================')
for train_iter in range(1):
    optimiser.zero_grad()
    for batch_row in range(5):
        inputs = 3.*poisson_input(0.5, 500, N)
        # targets = (torch.rand((10, N)) > 0.5).float()
        targets = feed_inputs_sequentially_return_spike_train(gen_model, inputs)
        spikes = feed_inputs_sequentially_return_spike_train(model, inputs)
        assert inputs.shape[0] == spikes.shape[0], "inputs.shape: {}, spikes.shape: {}".format(inputs.shape, spikes.shape)
        assert spikes.shape[0] == targets.shape[0]
        assert targets.sum() > 0.1 * 500, "assert targets spike in more than 10 % of time steps. targets.sum(): {}".format(targets.sum())
        assert spikes.sum() > 0.1 * 500, "assert spiking in more than 10 % of time steps. spikes.sum(): {}".format(spikes.sum())

        loss = van_rossum_dist(spikes=spikes, target_spikes=targets, tau=tensor(5.0))
        # loss = torch.sqrt(torch.pow(torch.sub(spikes, targets), 2).sum() + 1e-18)  # avoid sqrt(0) -> NaN
        print('loss: {}'.format(loss))
        print('num. of spikes: {}'.format(spikes.sum()))
        print('num. of target spikes: {}'.format(targets.sum()))

        loss.backward(retain_graph=True)
        optimiser.step()

        for param_i, param in enumerate(list(model.parameters())):
            # print('parameter #{}: {}'.format(param_i, param))
            print('parameter #{} gradient: {}'.format(param_i, param.grad))
            assert param.grad is not None and torch.abs(param.grad.sum()) > 1e-06, 'parameter #{} gradient: {}'.format(param_i, param.grad)

        model.reset_hidden_state()

params = {}
params[0] = [model.w[0].clone().detach().numpy()]
for row_i in range(1, model.w.shape[0]):
    params[0].append(model.w[row_i].clone().detach().numpy())

tar_params = {}
for row_j in range(gen_model.w.shape[0]):
    tar_params[row_j] = gen_model.w[row_j].clone().detach().numpy()

plot_all_param_pairs_with_variance(params, False, 'default', 'debug_file', 'model_params', 'Test weights plot (model)', False)
# plot_all_param_pairs_with_variance(tar_params, False, 'default', 'debug_file', 'gen_model_params', 'Test weights plot (gen_model)', False)
