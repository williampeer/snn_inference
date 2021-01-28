import os
import sys

import torch

from data_util import prefix, path
from spike_train_matlab_export import load_and_export_sim_data


def main(argv):
    print('Argument List:', str(argv))

    experiments_path = '/Users/william/repos/archives_snn_inference/archive 11/saved/'
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

        # exp_num_files = []
        for f in files:
            if f.__contains__('exp_num'):
                model_type = f.split('_exp_num_')[0]
                exp_num = int(f.split('_exp_num_')[1].split('_')[0])
                # exp_num_files.append(f)

                pdata_files = os.listdir(plot_data_path + folder_path)
                pdata_loss_files = []
                for pdata_f in pdata_files:
                    if pdata_f.__contains__('plot_losses'):
                        pdata_loss_files.append(pdata_f)

                pdata_loss_files.sort()
                cur_exp_pdata_loss_file = pdata_loss_files[exp_num]
                loss_data = torch.load(plot_data_path + folder_path + '/' + cur_exp_pdata_loss_file)
                custom_title = loss_data['plot_data']['custom_title']
                optimiser = custom_title.split(', ')[1].strip(' ')
                # model_type = custom_title.split(',')[0].split('(')[-1]
                lr = custom_title.split(', ')[-1].strip(' =lr').strip(')').replace('.', '')
                lfn = loss_data['plot_data']['fname'].split('loss_fn_')[1].split('_tau')[0]

                cur_fname = 'spikes_{}_{}_{}_{}_{}'.format(model_type, optimiser, lfn, lr, id)
                save_file_name = prefix + path + cur_fname + '.mat'
                if not os.path.exists(save_file_name):
                    load_and_export_sim_data(full_folder_path + f, fname=cur_fname)
                else:
                    print('file exists. skipping..')
                # load_and_export_sim_data(f, optim='Adam_frdvrda_001')

if __name__ == "__main__":
    main(sys.argv[1:])
