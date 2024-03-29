import sys

import numpy as np
import torch

from Dev.pytorch_custom_network_opt import in_place_cast_to_float32
from IO import save_model_params
from Models.GLIF import GLIF
from TargetModels import TargetEnsembleModels
from data_util import save_spiketrain_in_sparse_matlab_format, convert_to_sparse_vectors
from experiments import sine_modulated_white_noise_input
from model_util import generate_model_data


def main(argv):
    print('Argument List:', str(argv))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]

    t = 60 * 1000
    # t = 15 * 60 * 1000
    # model_path = 'glif1'
    # model = TargetModels.glif1()
    # poisson_rate = 0.4
    # model_path = '/home/william/repos/archives_snn_inference/archive (8)/saved/single_objective_optim/fitted_params_glif_ensembles_seed_1_optim_CMA_loss_fn_poisson_nll_budget_10000_exp_2.pt'
    model_path = '/home/william/repos/archives_snn_inference/archive (12)/saved/single_objective_optim/fitted_params_glif_ensembles_seed_1_optim_CMA_loss_fn_vrdfrd_10000_exp_5.pt'
    # model_path = '/home/william/repos/archives_snn_inference/archive (8)/saved/10-12_18-10-35-500/GLIF_exp_num_10_data_set_None_mean_loss_0.227_uuid_10-12_18-10-35-500.pt'
    # model_path = ''

    loss_fn = model_path.split('loss_fn_')[1].split('_budget')[0]
    cur_model_descr = model_path.split('fitted_params_')[1].split('_optim')[0]
    exp_num = model_path.split('exp_')[1].split('.pt')[0]
    optim = model_path.split('optim_')[1].split('_loss_fn')[0]
    lr = ''

    # loss_fn = 'firing_rate_distance_2'
    # cur_model_descr = 'glif_ensembles_seed_1'
    # exp_num = model_path.split('exp_num_')[1].split('_data_set')[0]
    # optim = 'Adam'
    # lr = '_lr_0_001'

    for i, opt in enumerate(opts):
        if opt == '-h':
            print('load_and_export_model_data.py -p <path> -t <time> -r <poisson-rate>')
            sys.exit()
        elif opt in ("-p", "--path"):
            model_path = args[i]
        elif opt in ("-t", "--time"):
            t = int(args[i])

    if model_path is None:
        print('No path to load model from specified.')
        sys.exit(1)

    recommended_params = torch.load(model_path)
    poisson_rate = recommended_params['rate']
    model_params = recommended_params.copy()
    del model_params['target_model'], model_params['target_rate'], model_params['time_interval'], model_params['loss_fn'], model_params['rate']
    model_params['preset_weights'] = model_params['w']
    in_place_cast_to_float32(model_params)
    model = GLIF(parameters=model_params)

    # exp_res = torch.load(model_path)
    # model = exp_res['model']
    # poisson_rate = exp_res['rate']

    # model = TargetEnsembleModels.glif_ensembles_model(1)
    # poisson_rate = 10.


    print('Loaded model.')

    interval_size = 4000
    interval_range = int(t/interval_size)
    assert interval_range > 0, "t must be greater than the interval size, {}. t={}".format(interval_size, t)

    spike_indices = np.array([], dtype='int8')
    spike_times = np.array([], dtype='float32')
    input_indices = np.array([], dtype='int8')
    input_times = np.array([], dtype='float32')
    print('Simulating data..')
    for t_i in range(interval_range):
        model.reset_hidden_state()
        # spiketrain = generate_synthetic_data(model, poisson_rate, t=interval_size)
        gen_input = sine_modulated_white_noise_input(rate=poisson_rate, t=interval_size, N=model.N)
        gen_spiketrain = generate_model_data(model=model, inputs=gen_input)
        # for gen spiketrain this may be thresholded to binary values:
        gen_spiketrain = torch.round(gen_spiketrain)

        inputs = gen_input.clone().detach()
        spiketrain = gen_spiketrain.clone().detach()

        # plot_spiketrain(spiketrain, 'Plot imported SNN', 'plot_imported_model')
        cur_input_indices, cur_input_times = convert_to_sparse_vectors(inputs, t_offset=t_i*interval_size)
        input_indices = np.append(input_indices, cur_input_indices)
        input_times = np.append(input_times, cur_input_times)
        cur_spike_indices, cur_spike_times = convert_to_sparse_vectors(spiketrain, t_offset=t_i*interval_size)
        spike_indices = np.append(spike_indices, cur_spike_indices)
        spike_times = np.append(spike_times, cur_spike_times)
        print('{} seconds ({:.2f} min) simulated.'.format(interval_size * (t_i+1)/1000., interval_size * (t_i+1)/(60.*1000)))

    # fname = model_path.split('/')[-1]
    # model_name = fname.split('.pt')[0]
    # save_fname_input = 'poisson_inputs_{}_t_{:.0f}s_rate_{}'.format(model_name, t/1000., poisson_rate).replace('.', '_') + '.mat'
    # save_spiketrain_in_sparse_matlab_format(fname=save_fname_input, spike_indices=input_indices, spike_times=input_times)
    # save_model_params(model, fname=save_fname_input.replace('.mat', '_params'))

    save_fname_output = 'fitted_spike_train_{}_{}_{}{}_exp_num_{}'.format(cur_model_descr, optim, loss_fn, lr, exp_num).replace('.', '_') + '.mat'
    # save_fname_output = 'glif_ensembles_seed_1.mat'
    save_spiketrain_in_sparse_matlab_format(fname=save_fname_output, spike_indices=spike_indices, spike_times=spike_times)
    save_model_params(model, fname=save_fname_output.replace('.mat', '_params'))


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
