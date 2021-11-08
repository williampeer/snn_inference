from Dev.setup_data_for_brian import *
from Dev.brian2_custom_network_opt import *
from TargetModels import TargetModels
from plot import plot_spike_trains_side_by_side

model_num = 0
target_params_dict = TargetModels.glif1().state_dict()
neuron_params = {}
for param_i, key in enumerate(target_params_dict):
    if key not in ['loss_fn', 'rate', 'w']:
        neuron_params[key] = target_params_dict[key].clone().detach().numpy()
# print(torch_model.state_dict()['w'])

time_interval = 4000
output_fname = output_fnames_rate_0_6[model_num]
output_data_path = target_data_path + output_fname
spike_node_indices, spike_times, spike_indices = data_util.load_sparse_data(output_data_path)
next_target_index_list = 0
next_target_index = 0

spike_train = get_spike_train_for(12., np.reshape(target_params_dict['w'].clone().detach().numpy(), (-1,)), neuron_params)

next_target_index, sample_targets = data_util.get_spike_train_matrix(index_last_step=0, advance_by_t_steps=time_interval,
                                                     spike_times=spike_times, spike_indices=spike_indices,
                                                     node_numbers=spike_node_indices)

print('brian2 spikes: {}, pytorch spikes: {}'.format(spike_train.sum(), sample_targets.sum()))

plot_spike_trains_side_by_side(spike_train, sample_targets, exp_type='single_objective_optim', uuid='TEST',
                               title='Spike trains brian test', fname='spike_trains_brian_test')
