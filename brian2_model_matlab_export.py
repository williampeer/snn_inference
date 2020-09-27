import torch

from Dev.brian2_custom_network_opt import get_spike_train_for_matlab_export
from IO import save_model_params
from data_util import save_spiketrain_in_sparse_matlab_format

dict_path = '/home/william/repos/archives_snn_inference/archive (2)/saved/single_objective_optim/fitted_params_optim_DE_loss_fn_vrdfrd_budget_3.pt'

optim_name = 'DE'
params_by_optim = torch.load(dict_path)[optim_name]
print('Loaded models dict.')

def convert_integer_indexed_to_named_params(d):
    parameter_names = ['E_L', 'C_m', 'G', 'R_I', 'f_v', 'f_I', 'delta_theta_s', 'b_s', 'a_v', 'b_v', 'theta_inf', 'delta_V', 'I_A']
    named_params_dict = {}
    for i in range(13):
        named_params_dict[parameter_names[i]] = d[i]
    return named_params_dict

model_parameters = convert_integer_indexed_to_named_params(params_by_optim)
for exp_i in range(4):
    print('Processing exp num {}'.format(exp_i))
    weights = params_by_optim['w'][exp_i]
    rate = params_by_optim['rate'][exp_i]
    current_model_parameters = {}
    for i, k in enumerate(model_parameters):
        current_model_parameters[k] = model_parameters[k][exp_i]

    brian_spikes = get_spike_train_for_matlab_export(rate, weights, current_model_parameters, run_time=4000)  # default 5m.
    print(brian_spikes)

    fname = dict_path.split('/')[-1]  # TODO: double check
    model_name = fname.split('.pt')[0]

    save_fname_output = 'spikes_brian_model_optim_{}_exp_{}_rate_{}'.format(optim_name, exp_i, rate).replace('.', '_') + '.mat'
    save_spiketrain_in_sparse_matlab_format(fname=save_fname_output, spike_indices=[], spike_times=spike_times)
    save_model_params(model, fname=save_fname_output.replace('.mat', '_params'))
