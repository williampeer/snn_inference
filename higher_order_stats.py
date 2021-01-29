import os
import sys

import numpy as np
import torch

import plot
import stats


def main(argv):
    print('Argument List:', str(argv))

    experiments_path = '/Users/william/repos/archives_snn_inference/archive 11/saved/plot_data/'
    folders = os.listdir(experiments_path)
    experiment_averages = {}
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

        if len(plot_spiketrains_files) != 55:
            # print("Incomplete exp. len should be 5 exp * 11 plots. was: {}".format(len(plot_spiketrains_files)))
            # print(len(plot_spiketrains_files))
            pass
        else:
            print('Processing final spike trains for configuration: {}, {}, {}, {}'.format(model_type, optimiser, lr, lfn))
            plot_spiketrains_files.sort()  # check that alphabetically

            if not experiment_averages.__contains__(model_type):
                experiment_averages[model_type] = {
                    optimiser: {lfn: {lr: {'corrcoeff': [], 'mu_model': [], 'std_model': [],
                                           'mu_target': [], 'std_target': [],
                                           'CV_model': [], 'CV_target': []}}}}
            if not experiment_averages[model_type].__contains__(optimiser):
                experiment_averages[model_type][optimiser] = {}
            if not experiment_averages[model_type][optimiser].__contains__(lfn):
                experiment_averages[model_type][optimiser][lfn] = {}
            if not experiment_averages[model_type][optimiser][lfn].__contains__(lr):
                experiment_averages[model_type][optimiser][lfn][lr] = {'corrcoeff': [], 'mu_model': [],
                                                                          'std_model': [],
                                                                          'mu_target': [], 'std_target': [],
                                                                          'CV_model': [], 'CV_target': []}

            # avg_rates_model = []; avg_rates_target = []
            corrcoeff_sum = None; mum = []; mut = []; stdm = []; stdt = []; CVm = []; CVt = []
            for exp_i in range(int(len(plot_spiketrains_files) / 11)):  # gen data for [0 + 11 * i]
                print('exp_i: {}'.format(exp_i))
                cur_full_path = full_folder_path + plot_spiketrains_files[11 * exp_i]

                data = torch.load(cur_full_path)
                plot_data = data['plot_data']
                model_spike_train = plot_data['model_spikes'].detach().numpy()
                target_spike_train = plot_data['target_spikes'].detach().numpy()

                plot.plot_spiketrains_side_by_side(torch.tensor(model_spike_train), torch.tensor(target_spike_train),
                                                   'export', model_type,
                                                   title='Final spike trains {}, {}, {}, $\\alpha={}$'.format(model_type, optimiser, lfn, lr),
                                                   fname='spike_train_{}_{}_{}_exp_{}'.format(model_type, optimiser, lfn, exp_i))

                corrcoeff, mu1, std1, mu2, std2, CV1, CV2 = stats.higher_order_stats(model_spike_train, target_spike_train, bin_size=100)


                cur_hyperconf = 'Correlation coefficient, {}, {}, {}, $\\alpha={}$'.format(model_type, optimiser, lfn, lr)
                fname_prefix = model_type + '_' + optimiser + '_' + lfn + '_' + lr

                id = cur_full_path.split('/')[-2]
                save_fname = '{}_{}_exp_num_{}.png'.format(fname_prefix, id, exp_i)

                plot.heatmap_spike_train_correlations(corrcoeff[12:, :12], axes=['Fitted model', 'Target model'],
                                                      exp_type='export', uuid=model_type+'/single_exp',
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

            avg_corrcoeff = (corrcoeff_sum / float(exp_i+1))[12:, :12]
            # print('avg_corrcoeff: {}'.format(avg_corrcoeff))
            for i in range(avg_corrcoeff.shape[0]):
                for j in range(avg_corrcoeff.shape[1]):
                    if np.isnan(avg_corrcoeff[i][j]):
                        avg_corrcoeff[i][j] = 0.
            cur_hyperconf = 'Average corrcoeff, {}, {}, {}, $\\alpha={}$'.format(model_type, optimiser, lfn, lr)
            plot.heatmap_spike_train_correlations(avg_corrcoeff, axes=['Fitted model', 'Target model'],
                                                  exp_type=plot_data['exp_type'], uuid='export',
                                                  fname='heatmap_bin_{}_avg_{}_exp_{}'.format(20, fname_prefix.replace('.', ''), id),
                                                  bin_size=20, custom_title=cur_hyperconf)

            experiment_averages[model_type][optimiser][lfn][lr]['corrcoeff'].append(np.copy(avg_corrcoeff))

            plot.bar_plot_pair_custom_labels(y1=mum, y2=mut,
                                             y1_std=stdm,
                                             y2_std=stdt,
                                             labels=False,
                                             exp_type='export', uuid=model_type,
                                             fname='bar_plot_avg_avg_{}'.format(
                                                 model_type + '_' + optimiser + '_' + lfn + '_' + lr).replace('.', ''),
                                             title='Average spike count within experiment', xlabel='Random seed')
            plot.bar_plot_pair_custom_labels(y1=CVm, y2=CVt,
                                             y1_std=np.std(CVm),
                                             y2_std=np.std(CVt),
                                             labels=False,
                                             exp_type='export', uuid=model_type, fname='bar_plot_avg_avg_CV_{}'.format(model_type + '_' + optimiser + '_' + lfn + '_' + lr).replace('.', ''),
                                             title='Avg. CV for spike count within experiment', xlabel='Random seed')

            experiment_averages[model_type][optimiser][lfn][lr]['mu_model'].append(np.mean(mum))
            experiment_averages[model_type][optimiser][lfn][lr]['std_model'].append(np.std(mum))
            experiment_averages[model_type][optimiser][lfn][lr]['mu_target'].append(np.mean(mut))
            experiment_averages[model_type][optimiser][lfn][lr]['std_target'].append(np.std(mut))
            experiment_averages[model_type][optimiser][lfn][lr]['CV_model'].append(np.mean(CVm))
            experiment_averages[model_type][optimiser][lfn][lr]['CV_target'].append(np.mean(CVt))

            # cur_std_model, cur_rate_model = stats.binned_avg_firing_rate_per_neuron(model_spike_train, bin_size=400)
            # cur_std_target, cur_rate_target = stats.binned_avg_firing_rate_per_neuron(target_spike_train, bin_size=400)

    plot_stats_across_experiments(experiment_averages)


def plot_stats_across_experiments(avg_statistics_per_exp):
    for m_i, m_k in enumerate(avg_statistics_per_exp):
        res_std_m = []; res_std_m_std = []
        res_std_t = []; res_std_t_std = []
        res_mu_m = []; res_mu_m_std = []
        res_mu_t = []; res_mu_t_std = []
        res_CV_m = []; res_CV_m_std = []
        res_CV_t = []; res_CV_t_std = []

        avg_diag_corrs = []
        avg_diag_corrs_std = []
        labels = []
        for o_i, o_k in enumerate(avg_statistics_per_exp[m_k]):
            for lfn_i, lfn_k in enumerate(avg_statistics_per_exp[m_k][o_k]):
                for lr_i, lr_k in enumerate(avg_statistics_per_exp[m_k][o_k][lfn_k]):
                    avg_stats_exps = avg_statistics_per_exp[m_k][o_k][lfn_k][lr_k]

                    res_std_m.append(np.mean(avg_stats_exps['std_model']))
                    res_std_t.append(np.mean(avg_stats_exps['std_target']))
                    res_std_m_std.append(np.std(avg_stats_exps['std_model']))
                    res_std_t_std.append(np.std(avg_stats_exps['std_target']))
                    res_mu_m.append(np.mean(avg_stats_exps['mu_model']))
                    res_mu_t.append(np.mean(avg_stats_exps['mu_target']))
                    res_mu_m_std.append(np.std(avg_stats_exps['mu_model']))
                    res_mu_t_std.append(np.std(avg_stats_exps['mu_target']))
                    res_CV_m.append(np.mean(avg_stats_exps['CV_model']))
                    res_CV_t.append(np.mean(avg_stats_exps['CV_target']))
                    res_CV_m_std.append(np.std(avg_stats_exps['CV_model']))
                    res_CV_t_std.append(np.std(avg_stats_exps['CV_target']))

                    corr_avgs = []
                    for c_i in range(len(avg_stats_exps['corrcoeff'])):
                        avg_diag_corr = (np.eye(12) * avg_stats_exps['corrcoeff'][0]).sum() / 12.
                        print('avg_diag_corr: {} for config ({}, {}, {}, {})'.format(avg_diag_corr, m_k, o_k, lfn_k, lr_k))
                        corr_avgs.append(avg_diag_corr)

                    avg_diag_corrs.append(np.mean(corr_avgs))
                    avg_diag_corrs_std.append(np.std(corr_avgs))

                    labels.append(o_k + ',\n' + lfn_k + ',\n$\\alpha$=' + lr_k)

        plot.bar_plot_pair_custom_labels(y1=res_mu_m, y2=res_mu_t, y1_std=res_mu_m_std, y2_std=res_mu_t_std, labels=labels,
                                         exp_type='export', uuid=m_k, fname='bar_plot_avg_mu_across_exp_{}'.format(m_k),
                                         title='Avg. spike count across experiments ({})'.format(m_k))
        plot.bar_plot_pair_custom_labels(y1=res_std_m, y2=res_std_t, y1_std=res_std_m_std, y2_std=res_std_t_std,
                                         labels=labels,
                                         exp_type='export', uuid=m_k, fname='bar_plot_avg_std_across_exp_{}'.format(m_k),
                                         title='Avg. spike standard deviation across experiments ({})'.format(m_k))
        plot.bar_plot_pair_custom_labels(y1=res_CV_m, y2=res_CV_t, y1_std=res_CV_m_std, y2_std=res_CV_t_std,
                                         labels=labels,
                                         exp_type='export', uuid=m_k, fname='bar_plot_avg_avg_CV_{}'.format(m_k),
                                         title='Avg. CV for spike count across experiments ({})'.format(m_k))

        baseline = 0.
        if m_k is 'LIF':
            baseline = 0.202
        elif m_k is 'GLIF':
            baseline = 0.325
        plot.bar_plot_crosscorrdiag(y1=avg_diag_corrs, y1_std=avg_diag_corrs_std, labels=labels,
                                         exp_type='export', uuid=m_k, fname='bar_plot_avg_diag_corrs_{}'.format(m_k),
                                         title='Avg. diag. corrs. across experiments ({})'.format(m_k), baseline=baseline)


if __name__ == "__main__":
    main(sys.argv[1:])
