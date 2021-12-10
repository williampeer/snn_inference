import numpy as np
import torch

import plot
import stats
from Models.no_grad.LIF_R_no_grad import LIF_R_no_grad
from TargetModels import TargetModels
from data_util import prefix, target_data_path
from experiments import sine_modulated_white_noise_input, draw_from_uniform
from model_util import generate_model_data


def plot_stats_across_experiments(avg_statistics_per_exp):
    for m_i, m_k in enumerate(avg_statistics_per_exp):
        # flat arrays per model type
        avg_model_rates = []
        stds_model_rates = []
        avg_target_rates = []
        stds_target_rates = []
        avg_model_rates_std = []
        avg_target_rates_std = []

        labels = []
        for o_i, o_k in enumerate(avg_statistics_per_exp[m_k]):
            avg_statistics_per_exp[m_k][o_k].pop('vrdfrda', None)
            for lfn_i, lfn_k in enumerate(avg_statistics_per_exp[m_k][o_k]):
                for lr_i, lr_k in enumerate(avg_statistics_per_exp[m_k][o_k][lfn_k]):
                    avg_stats_exps = avg_statistics_per_exp[m_k][o_k][lfn_k][lr_k]
                    print('processing: {}'.format(avg_stats_exps))

                    avg_model_rates.append(np.mean(avg_stats_exps['avg_model_rate']))
                    avg_model_rates_std.append(np.std(avg_stats_exps['avg_model_rate']))
                    avg_target_rates.append(np.mean(avg_stats_exps['avg_target_rate']))
                    avg_target_rates_std.append(np.std(avg_stats_exps['avg_target_rate']))

                    stds_model_rates.append(np.mean(avg_stats_exps['stds_model_rates']))
                    stds_target_rates.append(np.mean(avg_stats_exps['stds_target_rates']))
                    # stds_model_rates_std.append(np.std(avg_stats_exps['stds_model_rates']))
                    # stds_target_rates_std.append(np.std(avg_stats_exps['stds_target_rates']))

                    labels.append(o_k + ',\n' + lfn_k + ',\n$\\alpha$=' + lr_k)


        labels.append('{}\ninit\nmodels'.format(m_k))

        print('plotting for {}'.format(m_k))
        plot.bar_plot_pair_custom_labels(y1=avg_model_rates, y2=avg_target_rates, y1_std=avg_model_rates_std, y2_std=avg_target_rates_std,
                                         labels=labels,
                                         exp_type='export', uuid=m_k, fname='init_model_rate_bar_plot_avg_rate_across_exp_{}'.format(m_k),
                                         title='Avg. firing rates across experiments ({})'.format(m_k))
        plot.bar_plot_pair_custom_labels(y1=np.array(stds_model_rates)/np.array(avg_model_rates),
                                         y2=np.array(stds_target_rates)/np.array(avg_target_rates),
                                         y1_std=np.zeros_like(stds_model_rates),
                                         y2_std=np.zeros_like(stds_model_rates),
                                         labels=labels,
                                         exp_type='export', uuid=m_k, fname='init_model_rate_bar_plot_avg_rate_CV_{}.png'.format(m_k),
                                         title='Avg. CV for firing rate across experiments ({})'.format(m_k))


load_paths = []
offset = 42
num_neurons = 10
t_interval = 10000
experiment_averages = {}
avg_rates_model = []; stds_model_rates = []; avg_rates_target = []; stds_target_rates = []
for exp_i in range(4):
    non_overlapping_offset = offset + 3 + 1
    torch.manual_seed(non_overlapping_offset + exp_i)
    np.random.seed(non_overlapping_offset + exp_i)

    model_type = 'LIF_R'
    target_model = TargetModels.lif_r_continuous_ensembles_model_dales_compliant(random_seed=exp_i+3, N=num_neurons)

    optimiser = 'None'; lfn = 'None'; lr = 'None'

    exp_type = 'export'
    cur_fname = 'initial_spikes_LIF_R_exp_num_{}'.format(exp_i)
    save_file_name = prefix + target_data_path + cur_fname + '.mat'

    if not experiment_averages.__contains__(model_type):
        experiment_averages[model_type] = {
            optimiser: {lfn: {lr: {'avg_model_rate': [], 'stds_model_rates': [],
                                   'avg_target_rate': [], 'stds_target_rates': []}}}}
    if not experiment_averages[model_type].__contains__(optimiser):
        experiment_averages[model_type][optimiser] = {}
    if not experiment_averages[model_type][optimiser].__contains__(lfn):
        experiment_averages[model_type][optimiser][lfn] = {}
    if not experiment_averages[model_type][optimiser][lfn].__contains__(lr):
        experiment_averages[model_type][optimiser][lfn][lr] = {'avg_model_rate': [],
                                                               'stds_model_rates': [],
                                                               'avg_target_rate': [],
                                                               'stds_target_rates': []}

    print('exp_i: {}'.format(exp_i))

    init_params_model = draw_from_uniform(LIF_R_no_grad.parameter_init_intervals, N=num_neurons)
    neuron_types = np.ones((num_neurons,))
    for i in range(int(num_neurons / 3)):
        neuron_types[-(1 + i)] = -1
    model = LIF_R_no_grad(parameters=init_params_model, N=num_neurons, neuron_types=neuron_types)
    poisson_rate = 10.

    print('Loaded model.')

    model.reset_hidden_state()
    m_input = sine_modulated_white_noise_input(rate=poisson_rate, t=t_interval, N=model.N)
    m_spiketrain = generate_model_data(model=model, inputs=m_input)
    m_spiketrain = torch.round(m_spiketrain)

    target_model.reset_hidden_state()
    t_input = sine_modulated_white_noise_input(rate=poisson_rate, t=t_interval, N=target_model.N)
    t_spiketrain = generate_model_data(model=target_model, inputs=t_input)
    t_spiketrain = torch.round(t_spiketrain)

    cur_hyperconf = '{}, {}, {}, $\\alpha={}$'.format(model_type, optimiser, lfn, lr)
    fname_prefix = model_type + '_' + optimiser + '_' + lfn

    cur_rate_model = stats.mean_firing_rate(m_spiketrain).data.numpy()
    cur_rate_target = stats.mean_firing_rate(t_spiketrain).data.numpy()
    print('rate for current model spiketrain: {}'.format(cur_rate_model))
    print('rate for current target spiketrain: {}'.format(cur_rate_target))

    avg_model_rate = np.mean(np.asarray(cur_rate_model), axis=0) * 1000.
    avg_target_rate = np.mean(np.asarray(cur_rate_target), axis=0) * 1000.

    avg_rates_model.append(avg_model_rate)
    avg_rates_target.append(avg_target_rate)

    experiment_averages[model_type][optimiser][lfn][lr]['avg_model_rate'].append(np.mean(avg_rates_model))
    experiment_averages[model_type][optimiser][lfn][lr]['stds_model_rates'].append(np.std(avg_rates_model))
    experiment_averages[model_type][optimiser][lfn][lr]['avg_target_rate'].append(np.mean(avg_rates_target))
    experiment_averages[model_type][optimiser][lfn][lr]['stds_target_rates'].append(np.std(avg_rates_target))


plot_stats_across_experiments(avg_statistics_per_exp=experiment_averages)

# if __name__ == "__main__":
#     main(sys.argv[1:])
