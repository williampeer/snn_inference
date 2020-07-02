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

    t = 30 * 60 * 1000
    poisson_rate = 0.5
    # load_path = None
    # load_path = './saved/LIF_sleep_model/LIF_sleep_model.pt'
    # load_path = './saved/Izhikevich_sleep_model/Izhikevich_sleep_model.pt'
    # load_path ='/Users/william/repos/archives_snn_inference/archive inf 3006-1820/saved/06-30_15-53-19-119/LIF_exp_num_0_data_set_None_mean_loss_29.740_uuid_06-30_15-53-19-119.pt'
    # load_path ='/Users/william/repos/archives_snn_inference/archive inf 3006-1747/saved/06-30_14-58-39-186/LIF_exp_num_0_data_set_None_mean_loss_26.283_uuid_06-30_14-58-39-186.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive inf 0107-1313/saved/07-01_10-27-06-734/LIF_complex_exp_num_0_data_set_None_mean_loss_22.533_uuid_07-01_10-27-06-734.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive inf 0107-1619/saved/07-01_11-20-27-519/LIF_complex_exp_num_0_data_set_None_mean_loss_22.656_uuid_07-01_11-20-27-519.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive inf 0107-1619/saved/07-01_11-29-17-136/LIF_complex_exp_num_0_data_set_None_mean_loss_27.440_uuid_07-01_11-29-17-136.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive inf 0107-1937/saved/07-01_15-48-18-521/LIF_complex_exp_num_0_data_set_None_mean_loss_27.367_uuid_07-01_15-48-18-521.pt'
    # load_path = '/Users/william/repos/archives_snn_inference/archive inf 0107-morning/saved/06-30_19-07-48-687/LIF_exp_num_1_data_set_None_mean_loss_29.470_uuid_06-30_19-07-48-687.pt'
    load_path = '/Users/william/repos/archives_snn_inference/archive inf 0207-0750/saved/07-01_15-49-26-138/LIF_complex_exp_num_2_data_set_None_mean_loss_26.625_uuid_07-01_15-49-26-138.pt'

    for i, opt in enumerate(opts):
        if opt == '-h':
            print('load_and_export_model_data.py -p <path> -t <time> -r <poisson-rate>')
            sys.exit()
        elif opt in ("-p", "--path"):
            load_path = args[i]
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

    fname = load_path.split('/')[-1]
    model_name = fname.split('.pt')[0]
    save_fname = 'generated_spikes_{}_t_{:.1f}_s_rate_{}.mat'.format(model_name, t/1000., poisson_rate)
    save_spiketrain_in_matlab_format(fname=save_fname, spike_indices=spike_indices, spike_times=spike_times)


if __name__ == "__main__":
    main(sys.argv[1:])
