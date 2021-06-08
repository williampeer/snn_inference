import torch
import torch.nn as nn

from Models.LIF import LIF
from experiments import draw_from_uniform
from plot import plot_parameter_inference_trajectories_2d

num_neurons = 3
init_params_model = draw_from_uniform(LIF.parameter_init_intervals, num_neurons)
model = LIF(parameters=init_params_model, N=num_neurons, neuron_types=[1, 1, -1])
recovered_parameters = {}
target_parameters = {}
for param_i, param in enumerate(list(model.parameters())):
    print('parameter #{}: {}'.format(param_i, param))
    recovered_parameters[param_i] = [param.clone().detach().numpy()]
    target_parameters[param_i] = param.clone().detach().numpy()

model.E_L = nn.Parameter(-45. -20 * torch.rand(model.E_L.shape), requires_grad=True)
model.tau_m = nn.Parameter(1.5 + torch.rand(model.tau_m.shape), requires_grad=True)
model.tau_s = nn.Parameter(2.0 + 2. * torch.rand(model.tau_s.shape), requires_grad=True)
model.w = nn.Parameter(torch.rand(model.w.shape), requires_grad=True)
for param_i, param in enumerate(list(model.parameters())):
    print('parameter #{}: {}'.format(param_i, param))
    recovered_parameters[param_i].append(param.clone().detach().numpy())

model.E_L = nn.Parameter(-45. -20 * torch.rand(model.E_L.shape), requires_grad=True)
model.tau_m = nn.Parameter(1.5 + torch.rand(model.tau_m.shape), requires_grad=True)
model.tau_s = nn.Parameter(2.0 + 2. * torch.rand(model.tau_s.shape), requires_grad=True)
model.w = nn.Parameter(torch.rand(model.w.shape), requires_grad=True)
for param_i, param in enumerate(list(model.parameters())):
    print('parameter #{}: {}'.format(param_i, param))
    recovered_parameters[param_i].append(param.clone().detach().numpy())

model.E_L = nn.Parameter(-45. -20 * torch.rand(model.E_L.shape), requires_grad=True)
model.tau_m = nn.Parameter(1.5 + torch.rand(model.tau_m.shape), requires_grad=True)
model.tau_s = nn.Parameter(2.0 + 2. * torch.rand(model.tau_s.shape), requires_grad=True)
model.w = nn.Parameter(torch.rand(model.w.shape), requires_grad=True)
for param_i, param in enumerate(list(model.parameters())):
    print('parameter #{}: {}'.format(param_i, param))
    recovered_parameters[param_i].append(param.clone().detach().numpy())

# plot_all_param_pairs_with_variance(recovered_parameters,
#                                    uuid='test_uuid',
#                                    exp_type='default',
#                                    target_params=target_parameters,
#                                    custom_title="Average inferred parameters across experiments [{}, {}]".format('test', 'None'),
#                                    logger=False)

plot_parameter_inference_trajectories_2d(recovered_parameters, target_params=target_parameters, param_names=model.parameter_names,
                                         exp_type='default', uuid='test_trajectories', fname='test_plot_parameter_inference_trajectories_2d',
                                         custom_title='Test plot_parameter_inference_trajectories_2d', logger=False)

weights = recovered_parameters[0]
tar_weights = target_parameters[0]
assert len(weights[0].shape) == 2, "weights should be 2D"
weights_params = {}; tar_weights_params = {}; w_names = []
for n_i in range(model.N):
    weights_params[n_i] = weights[n_i]
    tar_weights_params[n_i] = tar_weights[n_i]
    w_names.append('w_{}'.format(n_i))

plot_parameter_inference_trajectories_2d(weights_params, target_params=tar_weights_params, param_names=w_names,
                                         exp_type='default', uuid='test_trajectories', fname='test_plot_parameter_inference_trajectories_weights_2d',
                                         custom_title='Test plot_parameter_inference_trajectories_weights_2d', logger=False)
