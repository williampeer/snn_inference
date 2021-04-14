import sys

import numpy as np
import torch

from IO import save_model_params
from TargetModels import TargetEnsembleModels
from data_util import save_spiketrain_in_sparse_matlab_format, convert_to_sparse_vectors
from experiments import poisson_input
from model_util import generate_model_data
from plot import plot_spike_train


def main(argv):
    print('Argument List:', str(argv))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]

    # model_type = None
    model_type = 'LIF_R'
    t = 5 * 60 * 1000
    # t = 15 * 60 * 1000
    for r_seed in range(4):
        poisson_rate = 10.
        model_path = ''

        for i, opt in enumerate(opts):
            if opt == '-h':
                print('load_and_export_model_data.py -p <path> -t <time> -r <poisson-rate>')
                sys.exit()
            elif opt in ("-mt", "--model-type"):
                model_type = str(args[i])
            elif opt in ("-t", "--time"):
                t = int(args[i])

        if model_type is not None:
            if model_type == 'LIF':
                model = TargetEnsembleModels.lif_ensembles_model_dales_compliant(random_seed=r_seed)
                model_name = 'lif_ensembles_dales_{}'.format(r_seed)
            elif model_type == 'LIF_R':
                model = TargetEnsembleModels.lif_r_ensembles_model_dales_compliant(random_seed=r_seed)
                model_name = 'lif_r_ensembles_dales_{}'.format(r_seed)
            elif model_type == 'GLIF':
                model = TargetEnsembleModels.glif_ensembles_model_dales_compliant(random_seed=r_seed)
                model_name = 'glif_ensembles_dales_{}'.format(r_seed)
            else:
                raise NotImplementedError('Model type not supported.')
        else:
            # model = TargetEnsembleModels.lif_ensembles_model_dales_compliant(random_seed=r_seed)
            # model_name = 'lif_ensembles_dales_{}'.format(r_seed)
            model = TargetEnsembleModels.glif_ensembles_model_dales_compliant(random_seed=r_seed)
            model_name = 'glif_ensembles_dales_{}'.format(r_seed)

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
            gen_input = poisson_input(rate=poisson_rate, t=interval_size, N=model.N)
            gen_spiketrain = generate_model_data(model=model, inputs=gen_input)

            gen_spiketrain = torch.round(gen_spiketrain)

            inputs = gen_input.clone().detach()
            spiketrain = gen_spiketrain.clone().detach()

            plot_spike_train(spiketrain, 'Plot imported SNN', 'plot_imported_model', fname='plot_'+model_name)
            cur_input_indices, cur_input_times = convert_to_sparse_vectors(inputs, t_offset=t_i*interval_size)
            input_indices = np.append(input_indices, cur_input_indices)
            input_times = np.append(input_times, cur_input_times)
            cur_spike_indices, cur_spike_times = convert_to_sparse_vectors(spiketrain, t_offset=t_i*interval_size)
            spike_indices = np.append(spike_indices, cur_spike_indices)
            spike_times = np.append(spike_times, cur_spike_times)
            print('{} seconds ({:.2f} min) simulated.'.format(interval_size * (t_i+1)/1000., interval_size * (t_i+1)/(60.*1000)))

        save_fname_output = 'matlab_export_{}'.format(model_name) + '.mat'
        save_spiketrain_in_sparse_matlab_format(fname=save_fname_output, spike_indices=spike_indices, spike_times=spike_times)
        save_model_params(model, fname=save_fname_output.replace('.mat', '_params'))


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
