import sys

import numpy as np
import torch

from IO import save_model_params
from TargetModels import TargetModels
from data_util import save_spiketrain_in_sparse_matlab_format, convert_to_sparse_vectors
from experiments import poisson_input
from model_util import generate_model_data


def main(argv):
    print('Argument List:', str(argv))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]

    t = 5 * 60 * 1000
    model_path = 'glif_slower_more_synchronous'
    model = TargetModels.glif_slower_more_synchronous()
    poisson_rate = 0.6
    # TODO: easy to program importing fnames.
    # model_path = '/home/william/repos/archives_snn_inference/archive (4)/saved/09-28_14-13-16-244/GLIF_exp_num_0_data_set_None_mean_loss_2.929_uuid_09-28_14-13-16-244.pt'
    # model_path = '/home/william/repos/archives_snn_inference/archive (4)/saved/09-28_14-13-16-244/GLIF_exp_num_2_data_set_None_mean_loss_2.997_uuid_09-28_14-13-16-244.pt'
    # model_path = '/home/william/repos/archives_snn_inference/archive (4)/saved/09-28_14-13-16-244/GLIF_exp_num_3_data_set_None_mean_loss_3.697_uuid_09-28_14-13-16-244.pt'
    # model_path = '/home/william/repos/archives_snn_inference/archive (4)/saved/09-28_14-13-16-244/GLIF_exp_num_4_data_set_None_mean_loss_2.982_uuid_09-28_14-13-16-244.pt'
    # model_path = '/home/william/repos/archives_snn_inference/archive (4)/saved/09-28_14-13-16-244/GLIF_exp_num_5_data_set_None_mean_loss_2.846_uuid_09-28_14-13-16-244.pt'
    # model_path = '/home/william/repos/archives_snn_inference/archive (4)/saved/09-28_14-13-16-244/GLIF_exp_num_6_data_set_None_mean_loss_6.513_uuid_09-28_14-13-16-244.pt'
    # model_path = '/home/william/repos/archives_snn_inference/archive (4)/saved/09-28_14-13-16-244/GLIF_exp_num_7_data_set_None_mean_loss_4.080_uuid_09-28_14-13-16-244.pt'
    # model_path = '/home/william/repos/archives_snn_inference/archive (4)/saved/09-28_14-13-16-244/GLIF_exp_num_8_data_set_None_mean_loss_3.252_uuid_09-28_14-13-16-244.pt'
    # model_path = '/home/william/repos/archives_snn_inference/archive (4)/saved/09-28_14-13-16-244/GLIF_exp_num_11_data_set_None_mean_loss_4.194_uuid_09-28_14-13-16-244.pt'
    # model_path = '/home/william/repos/archives_snn_inference/archive (4)/saved/09-28_14-13-16-244/GLIF_exp_num_14_data_set_None_mean_loss_3.004_uuid_09-28_14-13-16-244.pt'
    # model_path = '/home/william/repos/archives_snn_inference/archive (4)/saved/09-28_14-13-16-244/GLIF_exp_num_17_data_set_None_mean_loss_3.011_uuid_09-28_14-13-16-244.pt'
    # model_path = '/home/william/repos/archives_snn_inference/archive (4)/saved/09-28_14-13-16-244/GLIF_exp_num_18_data_set_None_mean_loss_2.879_uuid_09-28_14-13-16-244.pt'

    for i, opt in enumerate(opts):
        if opt == '-h':
            print('load_and_export_model_data.py -p <path> -t <time> -r <poisson-rate>')
            sys.exit()
        elif opt in ("-p", "--path"):
            model_path = args[i]
        elif opt in ("-t", "--time"):
            t = int(args[i])

    if model_path is None:
        print('No path to load model from specified.')
        sys.exit(1)

    # exp_res = torch.load(model_path)
    # model = exp_res['model']
    # poisson_rate = exp_res['rate']
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
        # spiketrain = generate_synthetic_data(model, poisson_rate, t=interval_size)
        gen_input = poisson_input(rate=poisson_rate, t=interval_size, N=model.N)
        _, gen_spiketrain = generate_model_data(model=model, inputs=gen_input)
        # for gen spiketrain this may be thresholded to binary values:
        gen_spiketrain = torch.round(gen_spiketrain)

        inputs = gen_input.clone().detach()
        spiketrain = gen_spiketrain.clone().detach()

        # plot_spiketrain(spiketrain, 'Plot imported SNN', 'plot_imported_model')
        cur_input_indices, cur_input_times = convert_to_sparse_vectors(inputs, t_offset=t_i*interval_size)
        input_indices = np.append(input_indices, cur_input_indices)
        input_times = np.append(input_times, cur_input_times)
        cur_spike_indices, cur_spike_times = convert_to_sparse_vectors(spiketrain, t_offset=t_i*interval_size)
        spike_indices = np.append(spike_indices, cur_spike_indices)
        spike_times = np.append(spike_times, cur_spike_times)
        print('{} seconds ({:.2f} min) simulated.'.format(interval_size * (t_i+1)/1000., interval_size * (t_i+1)/(60.*1000)))

    fname = model_path.split('/')[-1]
    model_name = fname.split('.pt')[0]
    # save_fname_input = 'poisson_inputs_{}_t_{:.0f}s_rate_{}'.format(model_name, t/1000., poisson_rate).replace('.', '_') + '.mat'
    # save_spiketrain_in_sparse_matlab_format(fname=save_fname_input, spike_indices=input_indices, spike_times=input_times)
    # save_model_params(model, fname=save_fname_input.replace('.mat', '_params'))

    save_fname_output = 'generated_spike_train_{}_t_{:.0f}s_rate_{:1.2f}'.format(model_name, t/1000., poisson_rate).replace('.', '_') + '.mat'
    save_spiketrain_in_sparse_matlab_format(fname=save_fname_output, spike_indices=spike_indices, spike_times=spike_times)
    save_model_params(model, fname=save_fname_output.replace('.mat', '_params'))


if __name__ == "__main__":
    main(sys.argv[1:])
