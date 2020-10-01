import torch

import Log
import data_util
from Dev.brian2_custom_network_opt import get_spike_train_for_matlab_export
from data_util import save_spiketrain_in_sparse_matlab_format, convert_brian_spike_train_to_matlab_format
from euclidean_similarity import mean_euclidean_similarity
from experiments import zip_dicts
from Dev.setup_data_for_brian import *
from geodesic_similarity import geodesic_similarity, mean_geodesic_similarity

# dict_path = '/home/william/repos/archives_snn_inference/archive (7)/saved/single_objective_optim/params_by_optim_optim_DE_loss_fn_vrdfrd_budget_2000.pt'
# dict_path = '/home/william/repos/archives_snn_inference/archive (7)/saved/single_objective_optim/params_by_optim_optim_DE_loss_fn_van_rossum_dist_budget_2000.pt'
# dict_path = '/home/william/repos/archives_snn_inference/archive (7)/saved/single_objective_optim/params_by_optim_optim_DE_loss_fn_poisson_nll_budget_2000.pt'
# dict_path = '/home/william/repos/archives_snn_inference/archive (7)/saved/single_objective_optim/params_by_optim_optim_DE_loss_fn_gamma_factor_budget_2000.pt'

# dict_path = '/home/william/repos/archives_snn_inference/archive (7)/saved/single_objective_optim/params_by_optim_optim_CMA_loss_fn_vrdfrd_budget_2000.pt'
# dict_path = '/home/william/repos/archives_snn_inference/archive (7)/saved/single_objective_optim/params_by_optim_optim_CMA_loss_fn_van_rossum_dist_budget_2000.pt'
dict_path = '/home/william/repos/archives_snn_inference/archive (7)/saved/single_objective_optim/params_by_optim_optim_CMA_loss_fn_poisson_nll_budget_2000.pt'

# optim_name = 'DE'
optim_name = dict_path.split('optim_optim_')[1].split('_loss_fn')[0]
params_by_optim = torch.load(dict_path)[optim_name]
print('Loaded models dict.')

fname = dict_path.split('/')[-1]
model_name = fname.split('.pt')[0]
logger = Log.Logger('euclid_similarity_brian_{}'.format(model_name))


def convert_integer_indexed_to_named_params(d):
    parameter_names = ['E_L', 'C_m', 'G', 'R_I', 'f_v', 'f_I', 'delta_theta_s', 'b_s', 'a_v', 'b_v', 'theta_inf', 'delta_V', 'I_A']
    named_params_dict = {}
    for i in range(13):
        named_params_dict[parameter_names[i]] = d[i]
    return named_params_dict


model_parameters = convert_integer_indexed_to_named_params(params_by_optim)

del target_params_dict['w']

for exp_i in range(len(model_parameters['E_L'])):
    print('Processing exp num {}'.format(exp_i))
    weights = params_by_optim['w'][exp_i]
    rate = params_by_optim['rate'][exp_i]
    loss_fn = params_by_optim['loss_fn'][exp_i]

    current_model_parameters = {}
    for i, k in enumerate(model_parameters):
        current_model_parameters[k] = model_parameters[k][exp_i]


    # save_fname_output = 'spikes_brian_{}_rate_{:2.2f}_exp_{}'.format(model_name, rate, exp_i).replace('.', '_') + '.mat'
    # save_fname_output = 'euclid_similarity_brian_{}_exp_{}'.format(model_name, exp_i).replace('.', '_')

    combined_model_params = {'w': weights, 'rate': rate}
    combined_model_params = zip_dicts(combined_model_params, current_model_parameters)

    print('euclid similarity exp #{}: {}'.format(exp_i, mean_euclidean_similarity(current_model_parameters, target_params_dict)))
    # print('mean geodesic similarity exp #{}: {}'.format(exp_i, mean_geodesic_similarity(current_model_parameters, target_params_dict)))

    # full_save_path = data_util.prefix + data_util.path + fname