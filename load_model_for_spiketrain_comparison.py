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

    t = 12000
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
    model.reset_hidden_state()
    sleep_model.reset_hidden_state()


    plot.plot_spiketrains_side_by_side(model_spiketrain, sleep_model_spiketrain, 'export',
                                       'default', 'Spike trains (Poisson input)',
                                       legend=['Fitted model', 'Sleep model'],
                                       fname='spike_trains_{}'.format(save_fname))

    s1 = model_spiketrain.detach().numpy()
    s2 = sleep_model_spiketrain.detach().numpy()

    bin_size = 400
    corrs = stats.spike_train_corr_new(s1=s1, s2=s2, bin_size=bin_size)
    # for i in range(corrs.shape[0]):
    #     for j in range(corrs.shape[1]):
    #         if(i==j):
    #             corrs[i,j]=0.1
    plot.heatmap_spike_train_correlations(corrs[12:, :12], axes=['Fitted model spike train', 'Sleep model spike train'],
                                          exp_type='default', uuid='export', fname='heatmap_bin_{}_{}'.format(bin_size, save_fname),
                                          bin_size=bin_size)
    plot.heatmap_spike_train_correlations(np.abs(corrs[12:, :12]), axes=['Fitted model spike train', 'Sleep model spike train'],
                                          exp_type='default', uuid='export',
                                          fname='heatmap_abs_bin_{}_{}'.format(bin_size, save_fname), bin_size=bin_size)

    std1, r1 = stats.binned_avg_firing_rate_per_neuron(s1, bin_size=bin_size)
    std2, r2 = stats.binned_avg_firing_rate_per_neuron(s2, bin_size=bin_size)
    plot.bar_plot_neuron_rates(r1, r2, std1, std2, bin_size=bin_size, exp_type='default', uuid='export',
                               fname='rate_plot_bin_{}_{}'.format(bin_size, save_fname))


if __name__ == "__main__":
    main(sys.argv[1:])
