import sys

import numpy as np
import torch

import IO
import data_util
from data_util import save_spiketrain_in_matlab_format, convert_to_sparse_vectors
from experiments import generate_synthetic_data


def main(argv):
    print('Argument List:', str(argv))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]

    # path = None
    # path = './Test/IzhikevichStable_sample.pt'
    folder = data_util.prefix + data_util.path
    fname = 'placeholder_'.format(IO.dt_descriptor())
    load_path = folder + fname
    t = 20 * 60 * 1000
    poisson_rate = 0.6

    for i, opt in enumerate(opts):
        if opt == '-h':
            print('load_and_export_model_data.py -mp <model-path> -fname <filename> -t <time> -r <poisson-rate>')
            sys.exit()
        elif opt in ("-mp", "--model-path"):
            load_path = args[i]
            fname = load_path.split('/')[-1]
        elif opt in ("-fname", "--filename"):
            fname = args[i]
        elif opt in ("-t", "--time"):
            t = int(args[i])
        elif opt in ("-r", "--poisson-rate"):
            poisson_rate = float(args[i])

    if load_path is None:
        print('No path to load model from specified.')
        sys.exit(1)

    model = torch.load(load_path)['model']
    print('Loaded model.')

    interval_size = 4000
    interval_range = int(t/interval_size)
    assert interval_range > 0, "t must be greater than the interval size, {}. t={}".format(interval_size, t)

    spike_indices = np.array([], dtype='int8')
    spike_times = np.array([], dtype='float32')
    print('Simulating data..')
    for t_i in range(interval_range):
        model.reset_hidden_state()
        spiketrain = generate_synthetic_data(model, poisson_rate, t=interval_size)
        # plot_spiketrain(spiketrain, 'Plot imported SNN', 'plot_imported_model')
        cur_spike_indices, cur_spike_times = convert_to_sparse_vectors(spiketrain, t_offset=t_i*interval_size)
        spike_indices = np.append(spike_indices, cur_spike_indices)
        spike_times = np.append(spike_times, cur_spike_times)
        print('{} seconds ({:.2f} min) simulated.'.format(interval_size * (t_i+1)/1000., interval_size * (t_i+1)/(60.*1000)))

    save_fname = 'generated_spikes_t_{:.1f}_s_rate_{}_'.format(t/1000., poisson_rate) + fname.split('.pt')[0] + '.mat'
    save_spiketrain_in_matlab_format(fname=save_fname, spike_indices=spike_indices, spike_times=spike_times)


if __name__ == "__main__":
    main(sys.argv[1:])