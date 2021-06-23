import os
import sys

import numpy as np
import torch

from Models.GLIF import GLIF
from Models.LIF import LIF
from Models.LIF_ASC import LIF_ASC
from Models.LIF_R import LIF_R
from Models.LIF_R_ASC import LIF_R_ASC
from TargetModels import TargetModels
from TargetModels.TargetModels import lif_continuous_ensembles_model_dales_compliant, \
    lif_r_continuous_ensembles_model_dales_compliant, lif_asc_continuous_ensembles_model_dales_compliant, \
    lif_r_asc_continuous_ensembles_model_dales_compliant, glif_continuous_ensembles_model_dales_compliant
from data_util import prefix, path
from experiments import draw_from_uniform
from spike_train_matlab_export import simulate_and_save_model_spike_train


def main(argv):
    num_neurons = 3
    duration = 15 * 60 * 1000
    glif_only_flag = False

    print('Argument List:', str(argv))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]
    for i, opt in enumerate(opts):
        if opt == '-h':
            print('main.py -N <num-neurons> -d <duration> -GF <glif-only-flag>')
            sys.exit()
        elif opt in ("-d", "--duration"):
            duration = int(args[i])
        elif opt in ("-N", "--num-neurons"):
            num_neurons = int(args[i])
        elif opt in ("-GF", "--glif-only-flag"):
            glif_only_flag = bool(args[i])

    for m_fn in [lif_continuous_ensembles_model_dales_compliant,
                 lif_r_continuous_ensembles_model_dales_compliant,
                 lif_asc_continuous_ensembles_model_dales_compliant,
                 lif_r_asc_continuous_ensembles_model_dales_compliant,
                 glif_continuous_ensembles_model_dales_compliant]:

        for f_i in range(3, 7):
            torch.manual_seed(f_i)
            np.random.seed(f_i)

            # init_params_model = draw_from_uniform(model_class.parameter_init_intervals, num_neurons)
            if not glif_only_flag:
                random_seed = f_i
                snn = m_fn(random_seed=random_seed, N=num_neurons)
            else:
                random_seed = 4
                snn = TargetModels.glif_continuous_ensembles_model_dales_compliant(random_seed=random_seed, N=num_neurons)

            cur_fname = 'target_model_spikes_{}_N_{}_seed_{}_duration_{}'.format(snn.name(), num_neurons, random_seed, duration)
            save_file_name = prefix + path + cur_fname + '.mat'
            if not os.path.exists(save_file_name):
                simulate_and_save_model_spike_train(model=snn, poisson_rate=10., t=duration, exp_num=f_i,
                                                    model_name=snn.name(), fname=cur_fname)
            else:
                print('file exists. skipping..')


if __name__ == "__main__":
    main(sys.argv[1:])
