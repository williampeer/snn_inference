import os
import sys

import numpy as np
import torch

import Constants
import Log
import data_util
from Constants import ExperimentType
from IO import makedir_if_not_exists
from data_util import prefix, path, load_sparse_data, get_spike_train_matrix
from eval import evaluate_loss
from exp_suite import stats_training_iterations


def sim_get_losses(model, node_indices, spike_times, spike_indices, t, p_rate, iterations, exp_num, constants, start_next_step):
    losses = np.asarray([])
    next_step = start_next_step
    for train_i in range(iterations):
        next_step, train_targets = get_spike_train_matrix(index_last_step=next_step,
                                                          advance_by_t_steps=t,
                                                          spike_times=spike_times, spike_indices=spike_indices,
                                                          node_numbers=node_indices)

        train_loss = evaluate_loss(model, inputs=None, p_rate=p_rate,
                                   target_spiketrain=train_targets, label='train i: {}'.format(train_i),
                                   exp_type=ExperimentType.DataDriven, train_i=train_i, exp_num=exp_num,
                                   constants=constants)
        # logger.log('pre-training loss:', parameters=['validation loss', validation_loss])
        print('training loss: {}'.format(['train loss', train_loss]))
        losses = np.concatenate((losses, np.asarray([train_loss])))
    return next_step, losses


def simulate_and_eval_loss(model, poisson_rate, rpti, exp_num, model_name, loss_fn, fname=False, uuid='Test_eval'):
    exp_num = int(exp_num)
    interval_size = rpti
    num_intervals = 5
    start_seed = 42
    non_overlapping_offset = start_seed
    torch.manual_seed(non_overlapping_offset + exp_num)
    np.random.seed(non_overlapping_offset + exp_num)  # reproduce randomness

    data_path = data_util.prefix + data_util.path + 'target_model_spikes_GLIF_N_3_seed_4_duration_1800000.mat'
    constants = Constants.Constants(0.005, train_iters=30, N_exp=3, batch_size=rpti, tau_van_rossum=10.0, initial_poisson_rate=poisson_rate,
                                    rows_per_train_iter=rpti, optimiser='SGD', loss_fn=loss_fn, evaluate_step=1, data_path=data_path,
                                    start_seed=start_seed, exp_type_str=ExperimentType.DataDriven.name)
    constants.UUID = uuid

    logger = Log.Logger('TestLog.txt')


    print('Simulating data..')
    # spike_train = None

    node_indices, spike_times, spike_indices = load_sparse_data(full_path=constants.data_path)
    next_step, test_targets = get_spike_train_matrix(index_last_step=0,
                                                      advance_by_t_steps=constants.rows_per_train_iter,
                                                      spike_times=spike_times, spike_indices=spike_indices,
                                                      node_numbers=node_indices)

    after_train_next_step, train_losses = sim_get_losses(model, node_indices, spike_times, spike_indices, t=interval_size,
                                                         p_rate=poisson_rate, iterations=num_intervals, exp_num=exp_num,
                                                         constants=constants, start_next_step=next_step)
    _, test_losses = sim_get_losses(model, node_indices, spike_times, spike_indices, t=interval_size, p_rate=poisson_rate,
                                    iterations=num_intervals, exp_num=exp_num, constants=constants,
                                    start_next_step=after_train_next_step)

    parameters = {}
    for p_i, key in enumerate(model.state_dict()):
        parameters[p_i] = [model.state_dict()[key].numpy()]
    stats_training_iterations(model_parameters=parameters, model=model, poisson_rate=poisson_rate,
                              train_losses=train_losses, test_losses=test_losses,
                              constants=constants, logger=logger, exp_type_str=constants.EXP_TYPE.name,
                              target_parameters=False, exp_num=exp_num, train_i=num_intervals)


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
                    save_file_name = prefix + path + archive_name + cur_fname + '.mat'

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
