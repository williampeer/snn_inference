import os
import sys

import numpy as np
import torch

import plot
import stats


def main(argv):
    print('Argument List:', str(argv))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]

    load_paths = []
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_spiketrains_side_by_side01-21_21-55-15-927.pt']
    experiments_path = '/Users/william/repos/archives_snn_inference/archive 13/saved/plot_data/'
    archive = '14_data'
    folders = os.listdir(experiments_path)
    loss_res = {}
    for folder_path in folders:
        # print(folder_path)
        full_folder_path = experiments_path + folder_path + '/'
        if not folder_path.__contains__('.DS_Store'):
            files = os.listdir(full_folder_path)
            id = folder_path.split('-')[-1]
        else:
            files = []
            id = 'None'
        plot_spiketrains_files = []
        plot_losses_files = []
        for f in files:
            if f.__contains__('plot_losses'):
                f_data = torch.load(full_folder_path + f)
                custom_title = f_data['plot_data']['custom_title']
                optimiser = custom_title.split(', ')[1].strip(' ')
                model_type = custom_title.split(',')[0].split('(')[-1]
                lr = custom_title.split(', ')[-1].strip(' =lr').strip(')')
                lfn = f_data['plot_data']['fname'].split('loss_fn_')[1].split('_tau')[0]

                plot_losses_files.append(f)

                # break
        if optimiser != 'SGD':
            pass
        else:
            # elif len(plot_losses_files) == 0:
            if len(plot_losses_files) == 0:
                print("Incomplete exp.: No loss files.")
                # print(len(plot_spiketrains_files))
                pass
            else:
                config = '{}_{}_{}_{}'.format(model_type, optimiser, lfn, lr.replace('.', '_'))
                print('Processing exp: {} (configuration: {})'.format(folder_path, config))
                plot_spiketrains_files.sort()  # check that alphabetically
                if not loss_res.__contains__(config):
                    loss_res[config] = {'train_loss': [], 'test_loss': []}
                    # loss_res[config]['train_loss'] = []
                    # loss_res[config]['test_loss'] = []

                # cur_hyperconf = '{}, {}, {}, $\\alpha={}$'.format(model_type, optimiser, lfn, lr)
                # fname_prefix = model_type + '_' + optimiser + '_' + lfn
                # save_fname = '{}_{}_train_iter_{}.eps'.format(fname_prefix, id, train_i)
                # custom_title = 'Average firing rates, '+cur_hyperconf+', {} iteration(s)'.format(train_i)

                train_losses = []; test_losses = []
                for loss_file in plot_losses_files:
                    data = torch.load(full_folder_path + loss_file)
                    print('Loaded saved plot data.')

                    plot_data = data['plot_data']
                    # plot_fn = data['plot_fn']
                    cur_train_loss = plot_data['training_loss']
                    cur_test_loss = plot_data['test_loss']

                    # diverged_outlier_loss = False
                    for l_i in range(len(cur_train_loss)):
                        max_val = 150.
                        if cur_train_loss[l_i] > max_val:
                            cur_train_loss[l_i] = max_val
                        if cur_test_loss[l_i] > max_val:
                            cur_test_loss[l_i] = max_val
                            # diverged_outlier_loss = True
                    # if not diverged_outlier_loss:
                    train_losses.append(cur_train_loss)
                    test_losses.append(cur_test_loss)

                avg_train_loss = np.mean(np.asarray(train_losses), axis=0)
                std_train_loss = np.std(np.asarray(train_losses), axis=0)
                avg_test_loss = np.mean(np.asarray(test_losses), axis=0)
                std_test_loss = np.std(np.asarray(test_losses), axis=0)

                loss_res[config]['train_loss'].append(avg_train_loss)
                loss_res[config]['test_loss'].append(avg_test_loss)

                # plot.plot_avg_losses(avg_train_loss, std_train_loss, avg_test_loss, std_test_loss, uuid='export',
                #                      custom_title=custom_title, fname='', custom_title='')

    conf_keys = list(loss_res.keys())
    conf_keys.sort()
    for conf in conf_keys:
        print('Plotting for conf: {}...'.format(conf))
        if len(loss_res[conf]['train_loss']) > 1:
            cur_avg_train_loss = np.mean(loss_res[conf]['train_loss'], axis=0)
            train_std = np.std(loss_res[conf]['train_loss'], axis=0)
            cur_avg_test_loss = np.mean(loss_res[conf]['test_loss'], axis=0)
            test_std = np.std(loss_res[conf]['test_loss'], axis=0)
            plot.plot_avg_losses(cur_avg_train_loss, train_std, cur_avg_test_loss, test_std, 'export/'+archive, exp_type=model_type,
                                 custom_title='Average loss across experiments, {}'.format(conf.replace('0_0', '0.0').replace('_', ', ')),
                                 fname='export_avg_loss_across_exp_{}.eps'.format(conf))

    # keys = ['LIF_Adam_frd_0_05', 'LIF_Adam_vrd_0_05', 'LIF_Adam_frdvrda_0_05']
    # keys = ['GLIF_Adam_frd_0_05', 'GLIF_Adam_vrd_0_05', 'GLIF_Adam_frdvrda_0_05']
    # plot.plot_avg_losses_composite(loss_res, keys)
    plot.plot_avg_losses_composite(loss_res, conf_keys)


if __name__ == "__main__":
    main(sys.argv[1:])
