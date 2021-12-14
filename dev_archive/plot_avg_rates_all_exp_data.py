import os

import numpy as np
import torch

import plot
import stats
from IO import makedir_if_not_exists
from TargetModels import TargetModels
from data_util import prefix, target_data_path
from experiments import sine_modulated_white_noise_input
from model_util import generate_model_data


def plot_stats_across_experiments(avg_statistics_per_exp):
    for m_i, m_k in enumerate(avg_statistics_per_exp):
        # flat arrays per model type
        avg_model_rates = []
        stds_model_rates = []
        avg_target_rates = []
        stds_target_rates = []
        avg_model_rates_std = []
        stds_model_rates_std = []
        avg_target_rates_std = []
        stds_target_rates_std = []

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
                    # stds_model_rates_std.append(np.std(avg_stats_exps['stds_model_rates']))
                    stds_target_rates.append(np.mean(avg_stats_exps['stds_target_rates']))
                    # stds_target_rates_std.append(np.std(avg_stats_exps['stds_target_rates']))

                    labels.append(o_k + ',\n' + lfn_k + ',\n$\\alpha$=' + lr_k)

        # init models
        # i_avg_rates_model, i_avg_rates_target, i_stds_model_rates, i_stds_target_rates = avg_rates_init_models(m_k)
        # avg_model_rates.append(np.mean(i_avg_rates_model))
        # avg_model_rates_std.append(np.std(i_avg_rates_model))
        # avg_target_rates.append(np.mean(i_avg_rates_target))
        # avg_target_rates_std.append(np.std(i_avg_rates_target))
        # stds_model_rates.append(np.mean(i_stds_model_rates))
        # stds_model_rates_std.append(np.std(i_stds_model_rates))
        # stds_target_rates.append(np.mean(i_stds_target_rates))
        # stds_target_rates_std.append(np.std(i_stds_target_rates))

        labels.append('{}\ninit\nmodels'.format(m_k))

        print('plotting for {}'.format(m_k))
        plot.bar_plot_pair_custom_labels(y1=avg_model_rates, y2=avg_target_rates, y1_std=avg_model_rates_std, y2_std=avg_target_rates_std,
                                         labels=labels,
                                         exp_type='export', uuid=m_k, fname='rate_bar_plot_avg_rate_across_exp_{}'.format(m_k),
                                         title='Avg. firing rates across experiments ({})'.format(m_k))
        # plot.bar_plot_pair_custom_labels(y1=stds_model_rates, y2=stds_target_rates_std, y1_std=stds_model_rates_std, y2_std=stds_target_rates_std,
        #                                  labels=labels,
        #                                  exp_type='export', uuid=m_k, fname='rate_bar_plot_avg_rate_std_across_exp_{}'.format(m_k),
        #                                  title='Avg. rate standard deviation across experiments ({})'.format(m_k))
        plot.bar_plot_pair_custom_labels(y1=np.array(stds_model_rates)/np.array(avg_model_rates),
                                         y2=np.array(stds_target_rates)/np.array(avg_target_rates),
                                         y1_std=np.zeros_like(stds_model_rates),
                                         y2_std=np.zeros_like(stds_model_rates),
                                         labels=labels,
                                         exp_type='export', uuid=m_k, fname='rate_bar_plot_avg_rate_CV_{}.png'.format(m_k),
                                         title='Avg. CV for firing rate across experiments ({})'.format(m_k))


# print('Argument List:', str(argv))
# opts = [opt for opt in argv if opt.startswith("-")]
# args = [arg for arg in argv if not arg.startswith("-")]

load_paths = []
# load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_spiketrains_side_by_side01-21_21-55-15-927.pt']
# experiments_path = '/Users/william/repos/archives_snn_inference/archive 10/saved/plot_data/'
# experiments_path = '/Users/william/repos/archives_snn_inference/archive 14/saved/plot_data/'
offset = 42
t_interval = 10000
experiments_path = '/home/william/repos/archives_snn_inference/archive/saved/'
archive_name = 'data/'
plot_data_path = experiments_path + 'plot_data/'
folders = os.listdir(experiments_path)
experiment_averages = {}
for folder_path in folders:
    full_folder_path = experiments_path + folder_path + '/'
    if not folder_path.__contains__('.DS_Store'):
        files = os.listdir(full_folder_path)
        id = folder_path.split('-')[-1]
    else:
        files = []
        id = 'None'

    avg_rates_model = []; stds_model_rates = []; avg_rates_target = []; stds_target_rates = []
    exp_i = 0
    for f in files:  # for exp_i in specific folder exp path
        model_type = f.split('_exp_num_')[0]

        if f.__contains__('exp_num') and model_type not in ['LIF', 'LIF_no_grad']:
            print('Success! Processing exp: {}'.format(f))

            exp_num = int(f.split('_exp_num_')[1].split('_')[0])

            pdata_files = os.listdir(plot_data_path + folder_path)
            pdata_loss_files = []
            for pdata_f in pdata_files:
                if pdata_f.__contains__('plot_losses'):
                    pdata_loss_files.append(pdata_f)

            pdata_loss_files.sort()
            if len(pdata_loss_files) > exp_num - offset:
                cur_exp_pdata_loss_file = pdata_loss_files[exp_num - offset]
                loss_data = torch.load(plot_data_path + folder_path + '/' + cur_exp_pdata_loss_file)
                custom_title = loss_data['plot_data']['custom_title']
                optimiser = custom_title.split(', ')[1].strip(' ')
                # model_type = custom_title.split(',')[0].split('(')[-1]
                lr = custom_title.split(', ')[-1].strip(' =lr').strip(')').replace('.', '')
                lfn = loss_data['plot_data']['fname'].split('loss_fn_')[1].split('_tau')[0]

                exp_type = 'DataDriven'
                cur_fname = 'spikes_{}_{}_{}_{}_{}_{}_exp_num_{}_60s'.format(exp_type, model_type, optimiser, lfn, lr,
                                                                             id, exp_num).replace('=', '_')
                save_file_name = prefix + target_data_path + archive_name + cur_fname + '.mat'

                if optimiser == 'SGD':
                    print('checking: {}'.format(save_file_name))

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

                    # ------------------------------------------------------------------------

                    print('exp_i: {}'.format(exp_i))

                    model_path = full_folder_path + f
                    cur_model_name = model_path.split('_exp_num')[0].split('/')[-1]
                    exp_num = model_path.split('exp_num_')[1].split('_data_set')[0]
                    # optim = model_path.split('optim_')[1].split('_loss_fn')[0]
                    # id = optim + '_' + model_path.split('.pt')[0].split('-')[-1]
                    # lr = ''

                    exp_res = torch.load(model_path)
                    model = exp_res['model']
                    poisson_rate = exp_res['rate']
                    # loss = data['loss']

                    print('Loaded model.')

                    model.reset_hidden_state()
                    m_input = sine_modulated_white_noise_input(rate=poisson_rate, t=t_interval, N=model.N)
                    m_spiketrain = generate_model_data(model=model, inputs=m_input)
                    m_spiketrain = torch.round(m_spiketrain)

                    target_model = TargetModels.lif_r_continuous_ensembles_model_dales_compliant(random_seed=exp_i + 3,
                                                                                                 N=model.N)
                    target_model.reset_hidden_state()
                    t_input = sine_modulated_white_noise_input(rate=poisson_rate, t=t_interval, N=target_model.N)
                    t_spiketrain = generate_model_data(model=target_model, inputs=t_input)
                    t_spiketrain = torch.round(t_spiketrain)


                    cur_hyperconf = '{}, {}, {}, $\\alpha={}$'.format(model_type, optimiser, lfn, lr)
                    fname_prefix = model_type + '_' + optimiser + '_' + lfn

                    # cur_std_model, cur_rate_model = stats.binned_avg_firing_rate_per_neuron(
                    #     m_spiketrain.detach().numpy(), bin_size=400)
                    cur_rate_model = stats.mean_firing_rate(m_spiketrain).data.numpy()
                    cur_rate_target = stats.mean_firing_rate(t_spiketrain).data.numpy()
                    print('rate for current spiketrain: {}'.format(cur_rate_model))
                    # cur_std_target, cur_rate_target = stats.binned_avg_firing_rate_per_neuron(
                    #     target_spikes['target_spikes'].detach().numpy(), bin_size=400)

                    avg_model_rate = np.mean(np.asarray(cur_rate_model), axis=0) * 1000.
                    # std_model_rates = np.mean(np.asarray(cur_std_model), axis=0) * 1000.
                    avg_target_rate = np.mean(np.asarray(cur_rate_target), axis=0) * 1000.
                    # std_target_rates = np.mean(np.asarray(cur_std_target), axis=0) * 1000.

                    avg_rates_model.append(avg_model_rate)
                    # stds_model_rates.append(std_model_rates)
                    avg_rates_target.append(avg_target_rate)
                    # stds_target_rates.append(std_target_rates)
                    # plot.plot_avg_losses(avg_train_loss, std_train_loss, avg_test_loss, std_test_loss, uuid='export',
                    #                      custom_title=custom_title, fname=save_fname)
                    # save_fname = '{}_{}_exp_num_{}.png'.format(fname_prefix, id, exp_i)
                    # custom_title = 'Average firing rates, '+cur_hyperconf+', 20 iteration(s)'
                    # plot.bar_plot_neuron_rates(avg_model_rate, avg_target_rate, std_model_rates, std_target_rates, bin_size=400,
                    #                            exp_type=plot_data['exp_type'], uuid='export',
                    #                            fname='export_rate_plot_{}'.format(save_fname), custom_title=custom_title)
                else:
                    print('Adam.. 💩')

                experiment_averages[model_type][optimiser][lfn][lr]['avg_model_rate'].append(np.mean(avg_rates_model))
                experiment_averages[model_type][optimiser][lfn][lr]['stds_model_rates'].append(np.std(avg_rates_model))
                experiment_averages[model_type][optimiser][lfn][lr]['avg_target_rate'].append(np.mean(avg_rates_target))
                experiment_averages[model_type][optimiser][lfn][lr]['stds_target_rates'].append(np.std(avg_rates_target))

                    # ------------------------------------------------------------------------
        exp_i += 1
        # ======================= ---------------------- !!!!!!!!!!!!!!!!!!!!! **********************
        # for exp_i in range(int(len(plot_spiketrains_files) / 21)):  # gen data for [0 + 11 * i]



plot_stats_across_experiments(avg_statistics_per_exp=experiment_averages)

# if __name__ == "__main__":
#     main(sys.argv[1:])
