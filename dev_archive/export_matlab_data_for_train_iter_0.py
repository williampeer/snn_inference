import os
import sys

import torch

from data_util import prefix, target_data_path
from spike_train_matlab_export import load_and_export_sim_data


def main(argv):
    print('Argument List:', str(argv))

    experiments_path = '/home/william/repos/archives_snn_inference/archive_1607/saved/'

    initial_models_path = 'initial_models'
    id = 'initial_model'
    for folder_path in [initial_models_path]:
        # print(folder_path)

        full_folder_path = experiments_path + folder_path + '/'
        if not folder_path.__contains__('.DS_Store'):
            files = os.listdir(full_folder_path)
        else:
            files = []

        # exp_num_files = []
        for f in files:
            if f.__contains__('exp_num'):
                model_type = f.split('_exp_num_')[0]
                exp_num = int(f.split('_exp_num_')[1].split('.pt')[0])
                cur_fname = 'spikes_{}_{}_exp_num_{}'.format(model_type, id, exp_num)
                save_file_name = prefix + target_data_path + cur_fname + '.mat'
                if not os.path.exists(save_file_name):
                    load_and_export_sim_data(full_folder_path + f, fname=cur_fname)
                else:
                    print('file exists. skipping..')


if __name__ == "__main__":
    main(sys.argv[1:])
