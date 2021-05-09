import os
import sys

import numpy as np
import torch

from Models.GLIF import GLIF
from Models.LIF import LIF
from Models.LIF_ASC import LIF_ASC
from Models.LIF_R import LIF_R
from Models.LIF_R_ASC import LIF_R_ASC
from TargetModels.TargetModels import lif_continuous_ensembles_model_dales_compliant, \
    lif_r_continuous_ensembles_model_dales_compliant, lif_asc_continuous_ensembles_model_dales_compliant, \
    lif_r_asc_continuous_ensembles_model_dales_compliant, glif_continuous_ensembles_model_dales_compliant
from data_util import prefix, path
from experiments import draw_from_uniform
from spike_train_matlab_export import simulate_and_save_model_spike_train


def main(argv):
    print('Argument List:', str(argv))

    for m_fn in [lif_continuous_ensembles_model_dales_compliant,
                 lif_r_continuous_ensembles_model_dales_compliant,
                 lif_asc_continuous_ensembles_model_dales_compliant,
                 lif_r_asc_continuous_ensembles_model_dales_compliant,
                 glif_continuous_ensembles_model_dales_compliant]:
        # model_name = model_class.__name__
        num_neurons = 12

        for f_i in range(3, 7):
            torch.manual_seed(f_i)
            np.random.seed(f_i)

            # init_params_model = draw_from_uniform(model_class.parameter_init_intervals, num_neurons)
            snn = m_fn(random_seed=f_i, N=num_neurons)

            cur_fname = 'target_model_spikes_{}_seed_{}'.format(snn.__class__.__name__, f_i)
            save_file_name = prefix + path + cur_fname + '.mat'
            if not os.path.exists(save_file_name):
                simulate_and_save_model_spike_train(model=snn, poisson_rate=10., t=60*1000, exp_num=f_i,
                                                    model_name=snn.__class__.__name__, fname=cur_fname)
            else:
                print('file exists. skipping..')


if __name__ == "__main__":
    main(sys.argv[1:])
