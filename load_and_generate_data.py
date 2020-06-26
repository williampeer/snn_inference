import sys

import numpy as np
import torch

import IO
from data_util import save_spiketrain_in_matlab_format, convert_to_sparse_vectors
from experiments import generate_synthetic_data
from plot import plot_spiketrain

# path = './Test/LIF_test.pt'
# path = './Test/IzhikevichStable_test.pt'


def main(argv):
    print('Argument List:', str(argv))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]

    # path = None
    # path = './Test/IzhikevichStable_sample.pt'
    path = './saved/IzhikevichStable_exp_num_0_mean_loss_41.887474060058594.pt'
    t = 20 * 60 * 1000
    poisson_rate = 0.6

    for i, opt in enumerate(opts):
        if opt == '-h':
            print('load_and_generate_data.py -p <model-path> -t <time> -r <poisson-rate>')
            sys.exit()
        elif opt in ("-p", "--model-path"):
            path = args[i]
        elif opt in ("-t", "--time"):
            t = int(args[i])
        elif opt in ("-r", "--poisson-rate"):
            poisson_rate = float(args[i])

    if path is None:
        print('No path to load model from specified.')
        sys.exit(1)

    model = torch.load(path)['model']

    interval_size = 4000
    interval_range = int(t/interval_size)

    spike_indices = np.array([], dtype='int8')
    spike_times = np.array([], dtype='float32')
    for t_i in range(interval_range):
        model.reset_hidden_state()
        spiketrain = generate_synthetic_data(model, poisson_rate, t=interval_size)
        # plot_spiketrain(spiketrain, 'Plot imported SNN', 'plot_imported_model')
        cur_spike_indices, cur_spike_times = convert_to_sparse_vectors(spiketrain, t_offset=t_i*interval_size)
        spike_indices = np.append(spike_indices, cur_spike_indices)
        spike_times = np.append(spike_times, cur_spike_times)
        print('Simulated a total of {} seconds of data'.format(interval_range * (t_i+1)))

    save_spiketrain_in_matlab_format(fname='model_spiketrain_{}.mat'.format(IO.dt_descriptor()), spike_indices=spike_indices, spike_times=spike_times)


if __name__ == "__main__":
    main(sys.argv[1:])
