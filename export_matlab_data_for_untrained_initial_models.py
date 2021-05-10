import os
import sys

import numpy as np
import torch

from Models.GLIF import GLIF
from Models.LIF import LIF
from Models.LIF_ASC import LIF_ASC
from Models.LIF_R import LIF_R
from Models.LIF_R_ASC import LIF_R_ASC
from data_util import prefix, path
from experiments import draw_from_uniform
from spike_train_matlab_export import simulate_and_save_model_spike_train


def main(argv):
    print('Argument List:', str(argv))

    for model_class in [LIF, LIF_ASC, LIF_R, LIF_R_ASC, GLIF]:
        model_name = model_class.__name__
        num_neurons = 12

        for exp_i in range(5):
            start_seed = 42
            non_overlapping_offset = start_seed + 5 + 1
            torch.manual_seed(non_overlapping_offset + exp_i)
            np.random.seed(non_overlapping_offset + exp_i)

            init_params_model = draw_from_uniform(model_class.parameter_init_intervals, num_neurons)
            snn = model_class(N=num_neurons, parameters=init_params_model,
                              neuron_types=[1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1])  # set to ground truth for this file only

            cur_fname = 'initial_model_spikes_{}_exp_num_{}_seed_{}_60s'.format(model_name, exp_i, non_overlapping_offset+exp_i)
            save_file_name = prefix + path + cur_fname + '.mat'
            if not os.path.exists(save_file_name):
                simulate_and_save_model_spike_train(model=snn, poisson_rate=10., t=60*1000, exp_num=exp_i,
                                                    model_name=model_name, fname=cur_fname)
            else:
                print('file exists. skipping..')


if __name__ == "__main__":
    main(sys.argv[1:])
