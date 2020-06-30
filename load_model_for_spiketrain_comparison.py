import sys

import numpy as np
import torch

import plot
import stats
from experiments import generate_synthetic_data


def main(argv):
    print('Argument List:', str(argv))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]

    t = 20000
    poisson_rate = 0.5
    # load_path = None
    sleep_model_path = './saved/LIF_sleep_model/LIF_sleep_model.pt'
    # load_path = './saved/Izhikevich_sleep_model/Izhikevich_sleep_model.pt'
    # load_path ='/Users/william/repos/archives_snn_inference/archive inf 3006-1820/saved/06-30_15-53-19-119/LIF_exp_num_0_data_set_None_mean_loss_29.740_uuid_06-30_15-53-19-119.pt'
    model_path ='/Users/william/repos/archives_snn_inference/archive inf 3006-1747/saved/06-30_14-58-39-186/LIF_exp_num_0_data_set_None_mean_loss_26.283_uuid_06-30_14-58-39-186.pt'
    save_fname = 'sleep_model_vs_{}_t_{}'.format(model_path.split('/')[-1].split('.pt')[0].replace('.', '_'), t)

    model = torch.load(model_path)['model']
    sleep_model = torch.load(sleep_model_path)['model']

    model_spiketrain = generate_synthetic_data(model, poisson_rate, t=t)
    sleep_model_spiketrain = generate_synthetic_data(sleep_model, poisson_rate, t=t)


    plot.plot_spiketrains_side_by_side(model_spiketrain, sleep_model_spiketrain, 'export',
                                       'default', 'Spike train fitted model versus sleep model (Poisson input)',
                                       fname='spike_trains_{}'.format(save_fname))

    s1 = model_spiketrain.detach().numpy()
    s2 = sleep_model_spiketrain.detach().numpy()
    corrs_400 = stats.spike_train_correlation(s1=s1, s2=s2, bin_size=400)
    plot.heatmap_spike_train_correlations(corrs_400, axes=['Fitted model', 'Slepe model'],
                                          exp_type='default', uuid='export',
                                          fname='heatmap_bin_400_{}'.format(save_fname))
    std1, r1 = stats.binned_avg_firing_rate_per_neuron(s1, bin_size=400)
    std2, r2 = stats.binned_avg_firing_rate_per_neuron(s2, bin_size=400)
    plot.bar_plot_neuron_rates(r1, r2, std1, std2, bin_size=400, exp_type='default', uuid='export',
                               fname='rate_plot_bin_400_{}'.format(save_fname))


if __name__ == "__main__":
    main(sys.argv[1:])
