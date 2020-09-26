from Dev.brian2_custom_network_opt import *
from TargetModels import TargetModels


# torch_model = TargetModels.glif1()
# print(torch_model.state_dict()['w'])
from plot import plot_spiketrains_side_by_side

neuron_params = {}
for param_i, key in enumerate(target_params_dict):
    if key not in ['loss_fn', 'rate', 'w']:
        neuron_params[key] = target_params_dict[key].clone().detach().numpy()
neuron_params['R_I'] = neuron_params['R_I']
spike_train = get_spike_train_for(20., np.reshape(target_params_dict['w'].clone().detach().numpy(), (-1,)), neuron_params)

# _, sample_targets = data_util.get_spike_train_matrix(index_last_step=0, advance_by_t_steps=time_interval,
#                                                      spike_times=spike_times, spike_indices=spike_indices,
#                                                      node_numbers=spike_node_indices)

plot_spiketrains_side_by_side(spike_train, torch.zeros_like(spike_train), exp_type='single_objective_optim', uuid='TEST',
                              title='Spike trains brian test', fname='spike_trains_brian_test')
