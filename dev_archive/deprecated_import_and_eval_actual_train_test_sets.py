import os
import sys

import numpy as np
import torch

import Constants
import Log
import data_util
from Constants import ExperimentType
from IO import makedir_if_not_exists
from data_util import prefix, target_data_path, load_sparse_data, get_spike_train_matrix
from eval import evaluate_loss
from exp_suite import stats_training_iterations


# TODO: Fix bug, train and test appear to be the same.
def simulate_and_eval_loss(model, poisson_rate, rpti, exp_num, model_name, loss_fn, fname=False, uuid='Test_eval'):
    exp_num = int(exp_num)
    interval_size = rpti
    num_intervals = 5
    start_seed = 42
    non_overlapping_offset = start_seed
    torch.manual_seed(non_overlapping_offset + exp_num)
    np.random.seed(non_overlapping_offset + exp_num)  # reproduce randomness

    data_path = data_util.prefix + data_util.target_data_path + 'target_model_spikes_GLIF_N_3_seed_4_duration_1800000.mat'
    constants = Constants.Constants(0.005, train_iters=30, N_exp=3, batch_size=12000, tau_van_rossum=10.0, initial_poisson_rate=poisson_rate,
                                    rows_per_train_iter=12000, optimiser='SGD', loss_fn=loss_fn, evaluate_step=1, data_path=data_path,
                                    start_seed=start_seed, exp_type_str=ExperimentType.DataDriven.name)

    logger = Log.Logger('TestLog.txt')


    print('Simulating data..')
    # spike_train = None

    node_indices, spike_times, spike_indices = load_sparse_data(full_path=constants.data_path)
    next_step, test_targets = get_spike_train_matrix(index_last_step=0,
                                                      advance_by_t_steps=constants.rows_per_train_iter,
                                                      spike_times=spike_times, spike_indices=spike_indices,
                                                      node_numbers=node_indices)

    train_losses = np.asarray([])
    test_losses = np.asarray([])
    for train_i in range(num_intervals):
        # if exp_type is ExperimentType.DataDriven.name:
        next_step, train_targets = get_spike_train_matrix(index_last_step=next_step,
                                                    advance_by_t_steps=constants.rows_per_train_iter,
                                                    spike_times=spike_times, spike_indices=spike_indices,
                                                    node_numbers=node_indices)

        train_loss = evaluate_loss(model, inputs=None, p_rate=poisson_rate,
                                        target_spiketrain=train_targets, label='train i: {}'.format(train_i),
                                        exp_type=constants.EXP_TYPE, train_i=train_i, exp_num=exp_num,
                                        constants=constants)
        # logger.log('pre-training loss:', parameters=['validation loss', validation_loss])
        print('training loss: {}'.format(['train loss', train_loss]))
        train_losses = np.concatenate((train_losses, np.asarray([train_loss])))

        next_step, test_targets = get_spike_train_matrix(index_last_step=next_step,
                                                    advance_by_t_steps=constants.rows_per_train_iter,
                                                    spike_times=spike_times, spike_indices=spike_indices,
                                                    node_numbers=node_indices)

        test_loss = evaluate_loss(model, inputs=None, p_rate=poisson_rate,
                                   target_spiketrain=test_targets, label='test i: {}'.format(train_i),
                                   exp_type=constants.EXP_TYPE, train_i=train_i, exp_num=exp_num,
                                   constants=constants)
        print('test loss: {}'.format(['test loss', test_loss]))
        test_losses = np.concatenate((test_losses, np.asarray([test_loss])))

        # cur_input_indices, cur_input_times = convert_to_sparse_vectors(inputs, t_offset=t_i * interval_size)
        # input_indices = np.append(input_indices, cur_input_indices)
        # input_times = np.append(input_times, cur_input_times)
        # cur_spike_indices, cur_spike_times = convert_to_sparse_vectors(spike_train, t_offset=t_i * interval_size)
        # spike_indices = np.append(spike_indices, cur_spike_indices)
        # spike_times = np.append(spike_times, cur_spike_times)
        print('{} seconds ({:.2f} min) simulated.'.format(interval_size * (train_i + 1) / 1000.,
                                                          interval_size * (train_i + 1) / (60. * 1000)))

    constants.UUID = uuid
    parameters = {}
    for p_i, key in enumerate(model.state_dict()):
        parameters[p_i] = [model.state_dict()[key].numpy()]
    stats_training_iterations(model_parameters=parameters, model=model, poisson_rate=poisson_rate,
                              train_losses=train_losses, test_losses=test_losses,
                              constants=constants, logger=logger, exp_type_str=constants.EXP_TYPE.name,
                              target_parameters=False, exp_num=exp_num, train_i=train_i)

    # fname = model_path.split('/')[-1]
    # model_name = fname.split('.pt')[0]
    # save_fname_input = 'poisson_inputs_{}_t_{:.0f}s_rate_{}'.format(model_name, t/1000., poisson_rate).replace('.', '_') + '.mat'
    # save_spiketrain_in_sparse_matlab_format(fname=save_fname_input, spike_indices=input_indices, spike_times=input_times)
    # save_model_params(model, fname=save_fname_input.replace('.mat', '_params'))
    # save_fname_output = 'fitted_spikes_{}_{}_{}_t_{}'.format(model_name, id, exp_num, t).replace('.', '_')
    # if not fname:
    #     fname = save_fname_output
    # if spike_train is not None:
    #     plot_spike_train(spike_train, 'Plot imported SNN', 'plot_imported_model', fname=fname)

    # save_fname_output = 'fitted_spike_train_{}_{}_{}{}_exp_num_{}'.format(cur_model_descr, optim, loss_fn, lr, exp_num).replace('.', '_') + '.mat'
    # save_fname_output = 'fitted_spike_train_{}_id_{}_exp_no_{}'.format(cur_model_name, id, exp_num).replace('.', '_') + '.mat'
    # save_spiketrain_in_sparse_matlab_format(fname=fname + '.mat', spike_indices=spike_indices, spike_times=spike_times)
    # save_model_params(model, fname=fname.replace('.mat', '_params'))


def load_model_eval_loss(model_path, lfn='FIRING_RATE_DIST', fname=False, rpti = 12000):
    # print('Argument List:', str(argv))

    # cur_model_descr = model_path.split('fitted_params_')[1].split('_optim')[0]
    cur_model_name = model_path.split('_exp_num')[0].split('/')[-1]
    exp_num = model_path.split('exp_num_')[1].split('_data_set')[0]
    # optim = model_path.split('optim_')[1].split('_loss_fn')[0]
    uuid = model_path.split('.pt')[0].split('-')[-1]
    # lr = ''

    exp_res = torch.load(model_path)
    model = exp_res['model']
    poisson_rate = exp_res['rate']
    # loss = data['loss']

    print('Loaded model.')

    simulate_and_eval_loss(model, poisson_rate, rpti, exp_num, cur_model_name, loss_fn=lfn, fname=fname, uuid=uuid)


def main(argv):
    print('Argument List:', str(argv))
    offset = 42

    experiments_path = '/home/william/repos/archives_snn_inference/archive (14)/saved/'
    archive_name = 'data/'
    plot_data_path = experiments_path + 'plot_data/'
    folders = os.listdir(experiments_path)
    # pdata_files = os.listdir(plot_data_path)

    spike_train_files_post_training = {}
    for folder_path in folders:
        # print(folder_path)

        full_folder_path = experiments_path + folder_path + '/'
        if not folder_path.__contains__('.DS_Store'):
            files = os.listdir(full_folder_path)
            id = folder_path.split('-')[-1]
        else:
            files = []
            id = 'None'

        for f in files:
            if f.__contains__('exp_num'):
                model_type = f.split('_exp_num_')[0]
                # if model_type not in ['LIF', 'LIF_R']:  # mt mask
                #     pass
                # else:
                exp_num = int(f.split('_exp_num_')[1].split('_')[0])

                pdata_files = os.listdir(plot_data_path + folder_path)
                pdata_loss_files = []
                for pdata_f in pdata_files:
                    if pdata_f.__contains__('plot_losses'):
                        pdata_loss_files.append(pdata_f)

                pdata_loss_files.sort()
                if len(pdata_loss_files) > exp_num-offset:
                    cur_exp_pdata_loss_file = pdata_loss_files[exp_num-offset]
                    loss_data = torch.load(plot_data_path + folder_path + '/' + cur_exp_pdata_loss_file)
                    custom_title = loss_data['plot_data']['custom_title']
                    optimiser = custom_title.split(', ')[1].strip(' ')
                    # model_type = custom_title.split(',')[0].split('(')[-1]
                    lr = custom_title.split(', ')[-1].strip(' =lr').strip(')').replace('.', '')
                    lfn = loss_data['plot_data']['fname'].split('loss_fn_')[1].split('_tau')[0]

                    exp_type = 'Synthetic'
                    cur_fname = 'loss_spiketrains_{}_{}_{}_{}_{}_{}_exp_num_{}'.format(exp_type, model_type, optimiser, lfn, lr, id, exp_num).replace('=', '_')
                    save_file_name = prefix + target_data_path + archive_name + cur_fname + '.mat'

                    if lfn.__contains__('FIRING_RATE_DIST'):
                        if optimiser == 'SGD':
                            print('checking: {}'.format(save_file_name))
                            # if os.path.exists(prefix + path + archive_name) and not os.path.exists(save_file_name):
                            makedir_if_not_exists('./figures/default/plot_imported_model/' + archive_name)
                            load_model_eval_loss(full_folder_path + f, lfn=lfn, fname=archive_name + cur_fname)
                            # else:
                            #     print('file exists. skipping..')
                        else:
                            print('Adam.. 💩')
                            # load_and_export_sim_data(f, optim='Adam_frdvrda_001')


if __name__ == "__main__":
    main(sys.argv[1:])
