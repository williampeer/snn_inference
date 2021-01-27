import os
import sys

import numpy as np
import torch

import plot
import stats


def main(argv):
    print('Argument List:', str(argv))

    experiments_path = '/Users/william/repos/archives_snn_inference/archive 10/saved/plot_data/'
    folders = os.listdir(experiments_path)
    avg_statistics_per_exp = {}
    for folder_path in folders:
        # print(folder_path)

        full_folder_path = experiments_path + folder_path + '/'
        if not folder_path.__contains__('.DS_Store'):
            files = os.listdir(full_folder_path)
            id = folder_path.split('-')[-1]
        else:
            files = []
            id = 'None'
        plot_spiketrains_files = []
        # plot_losses_files = []
        for f in files:
            if f.__contains__('plot_spiketrains_side_by_side'):
                plot_spiketrains_files.append(f)
            elif f.__contains__('plot_losses'):
                f_data = torch.load(full_folder_path + f)
                custom_title = f_data['plot_data']['custom_title']
                optimiser = custom_title.split(', ')[1].strip(' ')
                model_type = custom_title.split(',')[0].split('(')[-1]
                lr = custom_title.split(', ')[-1].strip(' =lr').strip(')')
                lfn = f_data['plot_data']['fname'].split('loss_fn_')[1].split('_tau')[0]
                # break

        # plot_losses_files.append(f)
        if not avg_statistics_per_exp.__contains__(model_type):
            avg_statistics_per_exp[model_type] = { optimiser: { lfn: { lr: { 'corrcoeff': [], 'mu_model': [], 'std_model': [],
                                                                                'mu_target': [], 'std_target': [],
                                                                                'CV_model': [], 'CV_target': []} } } }
        if not avg_statistics_per_exp[model_type].__contains__(optimiser):
            avg_statistics_per_exp[model_type][optimiser] = {}
        if not avg_statistics_per_exp[model_type][optimiser].__contains__(lfn):
            avg_statistics_per_exp[model_type][optimiser][lfn] = {}
        if not avg_statistics_per_exp[model_type][optimiser][lfn].__contains__(lr):
            avg_statistics_per_exp[model_type][optimiser][lfn][lr] = { 'corrcoeff': [], 'mu_model': [], 'std_model': [],
                                                                                'mu_target': [], 'std_target': [],
                                                                                'CV_model': [], 'CV_target': []}

        # cur_hyperconf = '{}, {}, {}, $\\alpha={}$'.format(model_type, optimiser, lfn, lr)
        # fname_prefix = model_type + '_' + optimiser + '_' + lfn + '_' + lr
        # save_fname = '{}_{}_train_iter_{}.eps'.format(fname_prefix, id, 20)
        # custom_title = 'Average firing rates, '+cur_hyperconf+', {} iteration(s)'.format(20)

        if len(plot_spiketrains_files) != 55:
            # print("Incomplete exp. len should be 5 exp * 11 plots. was: {}".format(len(plot_spiketrains_files)))
            # print(len(plot_spiketrains_files))
            pass
        else:
            print('Processing final spike trains for configuration: {}, {}, {}, {}'.format(model_type, optimiser, lr, lfn))
            plot_spiketrains_files.sort()  # check that alphabetically

            # avg_rates_model = []; avg_rates_target = []
            corrcoeff_sum = None; mum = []; mut = []; stdm = []; stdt = []; CVm = []; CVt = []
            for exp_i in range(int(len(plot_spiketrains_files) / 11)):
                # gen data for [0 + 11 * i]
                # loss_res[model_type][optimiser][lr][lfn] = loss_res[model_type][optimiser][lr][lfn] + plot_spiketrains_files
                # print('Loaded saved plot data.')
                # plot_data = data['plot_data']
                cur_full_path = full_folder_path + plot_spiketrains_files[11 * exp_i]

                data = torch.load(cur_full_path)
                plot_data = data['plot_data']
                model_spike_train = plot_data['model_spikes'].detach().numpy()
                target_spike_train = plot_data['target_spikes'].detach().numpy()

                # corrcoef = stats.spike_train_corr_new(model_spike_train, target_spike_train)
                corrcoeff, mu1, std1, mu2, std2, CV1, CV2 = stats.higher_order_stats(model_spike_train, target_spike_train)

                cur_hyperconf = 'Correlation coefficient, {}, {}, {}, $\\alpha={}$'.format(model_type, optimiser, lfn, lr)
                fname_prefix = model_type + '_' + optimiser + '_' + lfn + '_' + lr

                id = cur_full_path.split('/')[-2]
                save_fname = '{}_{}_exp_num_{}.eps'.format(fname_prefix, id, exp_i)

                plot.heatmap_spike_train_correlations(corrcoeff[12:, :12], axes=['Fitted model', 'Target model'],
                                                      exp_type=plot_data['exp_type'], uuid='export',
                                                      fname='heatmap_bin_{}_{}'.format(20, save_fname),
                                                      bin_size=20, custom_title=cur_hyperconf)

                if corrcoeff_sum is None:
                    corrcoeff_sum = np.zeros_like(corrcoeff) + corrcoeff
                else:
                    corrcoeff_sum = corrcoeff_sum + corrcoeff
                mum.append(mu1)
                mut.append(mu2)
                stdm.append(std1)
                stdt.append(std2)
                CVm.append(CV1)
                CVt.append(CV2)

            avg_corrcoeff = corrcoeff_sum / float(exp_i+1)
            cur_hyperconf = 'Average corrcoeff, {}, {}, {}, $\\alpha={}$'.format(model_type, optimiser, lfn, lr)
            plot.heatmap_spike_train_correlations(avg_corrcoeff[12:, :12], axes=['Fitted model', 'Target model'],
                                                  exp_type=plot_data['exp_type'], uuid='export',
                                                  fname='heatmap_bin_{}_avg_{}_exp_{}'.format(20, fname_prefix.replace('.', ''), id),
                                                  bin_size=20, custom_title=cur_hyperconf)

            avg_statistics_per_exp[model_type][optimiser][lfn][lr]['corrcoeff'].append(avg_corrcoeff)

            avg_spike_count_model = np.mean(mum)
            avg_spike_count_std_model = np.mean(stdm)
            avg_spike_count_target = np.mean(mut)
            avg_spike_count_std_target = np.mean(stdt)
            CV_model = np.mean(CVm)
            CV_target = np.mean(CVt)

            avg_statistics_per_exp[model_type][optimiser][lfn][lr]['mu_model'].append(avg_spike_count_model)
            avg_statistics_per_exp[model_type][optimiser][lfn][lr]['std_model'].append(avg_spike_count_std_model)
            avg_statistics_per_exp[model_type][optimiser][lfn][lr]['mu_target'].append(avg_spike_count_target)
            avg_statistics_per_exp[model_type][optimiser][lfn][lr]['std_target'].append(avg_spike_count_std_target)
            avg_statistics_per_exp[model_type][optimiser][lfn][lr]['CV_model'].append(CV_model)
            avg_statistics_per_exp[model_type][optimiser][lfn][lr]['CV_target'].append(CV_target)

            # cur_std_model, cur_rate_model = stats.binned_avg_firing_rate_per_neuron(model_spike_train, bin_size=400)
            # cur_std_target, cur_rate_target = stats.binned_avg_firing_rate_per_neuron(target_spike_train, bin_size=400)

        for m_i, m_k in enumerate(avg_statistics_per_exp):
            res_std_m = []
            res_std_t = []
            res_mu_m = []
            res_mu_t = []
            res_CV_m = []; res_CV_m_std = []
            res_CV_t = []; res_CV_t_std = []
            labels = []
            for o_i, o_k in enumerate(avg_statistics_per_exp[m_k]):
                for lfn_i, lfn_k in enumerate(avg_statistics_per_exp[m_k][o_k]):
                    for lr_i, lr_k in enumerate(avg_statistics_per_exp[m_k][o_k][lfn_k]):
                        avg_stats_exps = avg_statistics_per_exp[m_k][o_k][lfn_k][lr_k]

                        avg_avg_mu_model = np.mean(avg_stats_exps['mu_model'])
                        avg_avg_mu_target = np.mean(avg_stats_exps['mu_target'])
                        avg_avg_std_model = np.mean(avg_stats_exps['std_model'])
                        avg_avg_std_target = np.mean(avg_stats_exps['std_target'])
                        avg_avg_CV_model = np.mean(avg_stats_exps['CV_model'])
                        avg_avg_CV_target = np.mean(avg_stats_exps['CV_target'])

                        plot.bar_plot_pair_custom_labels(y1=avg_stats_exps['mu_model'], y2=avg_stats_exps['mu_target'],
                                                         y1_std=avg_stats_exps['std_model'], y2_std=avg_stats_exps['std_target'],
                                                         labels=labels,
                                                         exp_type='export', uuid=m_k,
                                                         fname='bar_plot_avg_avg_{}'.format(m_k+'_'+o_k+'_'+lfn_k+'_'+lr_k),
                                                         title='Average spike count and variance for experiment')
                        plot.bar_plot_pair_custom_labels(y1=avg_stats_exps['CV_model'], y2=avg_stats_exps['CV_target'],
                                                         y1_std=np.std(avg_stats_exps['CV_model']), y2_std=np.std(avg_stats_exps['CV_target']),
                                                         labels=labels,
                                                         exp_type='export', uuid=m_k, fname='bar_plot_avg_avg_CV',
                                                         title='Average coefficient of variation of spike count across generative models and experiments')

                        res_std_m.append(avg_avg_std_model)
                        res_std_t.append(avg_avg_std_target)
                        res_mu_m.append(avg_avg_mu_model)
                        res_mu_t.append(avg_avg_mu_target)
                        res_CV_m.append(avg_avg_CV_model)
                        res_CV_t.append(avg_avg_CV_target)
                        res_CV_m_std.append(np.std(avg_stats_exps['CV_model']))
                        res_CV_t_std.append(np.std(avg_stats_exps['CV_target']))

                        labels.append(o_k + ', ' + lfn_k + ',\n$\\alpha$=' + lr_k)

            plot.bar_plot_pair_custom_labels(y1=res_mu_m, y2=res_mu_t, y1_std=res_std_m, y2_std=res_std_t, labels=labels,
                                             exp_type='export', uuid=m_k, fname='bar_plot_avg_avg_mu', title='Average spike count and variance for generative models')
            plot.bar_plot_pair_custom_labels(y1=res_CV_m, y2=res_CV_t, y1_std=res_CV_m_std, y2_std=res_CV_t_std, labels=labels,
                                             exp_type='export', uuid=m_k, fname='bar_plot_avg_avg_CV',
                                             title='Average coefficient of variation of spike count for generative models')


if __name__ == "__main__":
    main(sys.argv[1:])
