import os
import sys

import numpy as np
import torch

import plot
import stats


def main(argv):
    print('Argument List:', str(argv))

    init_LIF_models = '/Users/william/repos/snn_inference/saved/plot_data/init_LIF_models/'
    init_GLIF_models = '/Users/william/repos/snn_inference/saved/plot_data/init_GLIF_models/'
    avg_statistics_per_exp = {}
    for full_folder_path in [init_LIF_models, init_GLIF_models]:
        print(full_folder_path)

        if not full_folder_path.__contains__('.DS_Store'):
            files = os.listdir(full_folder_path)
            id = full_folder_path.split('/')[-1]
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

        print('Processing final spike trains for configuration: {}, {}, {}, {}'.format(model_type, optimiser, lr, lfn))
        plot_spiketrains_files.sort()  # check that alphabetically

        if not avg_statistics_per_exp.__contains__(model_type):
            avg_statistics_per_exp[model_type] = {
                optimiser: {lfn: {lr: {'corrcoeff': [], 'mu_model': [], 'std_model': [],
                                       'mu_target': [], 'std_target': [],
                                       'CV_model': [], 'CV_target': []}}}}
        if not avg_statistics_per_exp[model_type].__contains__(optimiser):
            avg_statistics_per_exp[model_type][optimiser] = {}
        if not avg_statistics_per_exp[model_type][optimiser].__contains__(lfn):
            avg_statistics_per_exp[model_type][optimiser][lfn] = {}
        if not avg_statistics_per_exp[model_type][optimiser][lfn].__contains__(lr):
            avg_statistics_per_exp[model_type][optimiser][lfn][lr] = {'corrcoeff': [], 'mu_model': [],
                                                                      'std_model': [],
                                                                      'mu_target': [], 'std_target': [],
                                                                      'CV_model': [], 'CV_target': []}

        # avg_rates_model = []; avg_rates_target = []
        corrcoeff_sum = None; mum = []; mut = []; stdm = []; stdt = []; CVm = []; CVt = []
        for exp_i in range(len(plot_spiketrains_files)):
            print('exp_i: {}'.format(exp_i))
            cur_full_path = full_folder_path + plot_spiketrains_files[exp_i]

            data = torch.load(cur_full_path)
            plot_data = data['plot_data']
            model_spike_train = plot_data['model_spikes'].detach().numpy()
            target_spike_train = plot_data['target_spikes'].detach().numpy()

            plot.plot_spiketrains_side_by_side(torch.tensor(model_spike_train), torch.tensor(target_spike_train),
                                               'export', model_type,
                                               title='Final spike trains {}, {}, {}, $\\alpha={}$'.format(model_type, optimiser, lfn, lr),
                                               fname='initial_spike_train_{}_{}_{}_exp_{}.eps'.format(model_type, optimiser, lfn, exp_i))

            corrcoeff, mu1, std1, mu2, std2, CV1, CV2 = stats.higher_order_stats(model_spike_train, target_spike_train, bin_size=100)


            cur_hyperconf = 'Correlation coefficient, {}, {}, {}, $\\alpha={}$'.format(model_type, optimiser, lfn, lr)
            fname_prefix = model_type + '_' + optimiser + '_' + lfn + '_' + lr

            id = cur_full_path.split('/')[-2]
            save_fname = '{}_{}_exp_num_{}.eps'.format(fname_prefix, id, exp_i)

            plot.heatmap_spike_train_correlations(corrcoeff[12:, :12], axes=['Fitted model', 'Target model'],
                                                  exp_type='export', uuid=model_type+'/single_exp',
                                                  fname='initial_heatmap_bin_{}_{}.eps'.format(20, save_fname),
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
                                              fname='initial_heatmap_bin_{}_avg_{}_exp_{}.eps'.format(20, fname_prefix.replace('.', ''), id),
                                              bin_size=20, custom_title=cur_hyperconf)

        plot.bar_plot_pair_custom_labels(y1=mum, y2=mut,
                                         y1_std=stdm,
                                         y2_std=stdt,
                                         labels=False,
                                         exp_type='export', uuid=model_type,
                                         fname='initial_bar_plot_avg_avg_{}.eps'.format(
                                             model_type + '_' + optimiser + '_' + lfn + '_' + lr).replace('.', ''),
                                         title='Average spike count within experiment', xlabel='Random seed',
                                         legend=['Initial model', 'Target model'])
        plot.bar_plot_pair_custom_labels(y1=CVm, y2=CVt,
                                         y1_std=np.std(CVm),
                                         y2_std=np.std(CVt),
                                         labels=False,
                                         exp_type='export', uuid=model_type, fname='initial_bar_plot_avg_avg_CV_{}.eps'.format(model_type + '_' + optimiser + '_' + lfn + '_' + lr).replace('.', ''),
                                         title='Avg. CV for spike count within experiment', xlabel='Random seed',
                                         legend=['Initial model', 'Target model'])

        plot.bar_plot_pair_custom_labels(y1=[np.mean(mum)], y2=[np.mean(mut)], y1_std=[np.std(mum)], y2_std=[np.std(mut)], labels=[optimiser + ',\n' + lfn + ',\n$\\alpha$=' + lr],
                                         exp_type='export', uuid=model_type, fname='initial_bar_plot_avg_mu_across_exp_{}.eps'.format(model_type),
                                         title='Avg. spike count across experiments ({})'.format(model_type),
                                         legend=['Initial model', 'Target model'])

        plot.bar_plot_pair_custom_labels(y1=[np.mean(stdm)], y2=[np.mean(stdt)], y1_std=[np.std(stdm)], y2_std=[np.std(stdt)],
                                         labels=[optimiser + ',\n' + lfn + ',\n$\\alpha$=' + lr],
                                         exp_type='export', uuid=model_type, fname='initial_bar_plot_avg_std_across_exp_{}.eps'.format(model_type),
                                         title='Avg. spike standard deviation across experiments ({})'.format(model_type),
                                         legend=['Initial model', 'Target model'])

        plot.bar_plot_pair_custom_labels(y1=[np.mean(CVm)], y2=[np.mean(CVt)], y1_std=0., y2_std=0.,
                                         labels=[optimiser + ',\n' + lfn + ',\n$\\alpha$=' + lr],
                                         exp_type='export', uuid=model_type, fname='initial_bar_plot_avg_avg_CV_{}.eps'.format(model_type),
                                         title='Avg. CV for spike count across experiments ({})'.format(model_type),
                                         legend=['Initial model', 'Target model'])

        mean_avg_corrcoeff = (np.eye(12) * avg_corrcoeff[12:, :12]).sum() / 12.
        print('diag mean avg_corrcoeff: {}'.format(mean_avg_corrcoeff))

        return mum, stdm, CVm, mut, stdt, CVt, mean_avg_corrcoeff
        # cur_std_model, cur_rate_model = stats.binned_avg_firing_rate_per_neuron(model_spike_train, bin_size=400)
        # cur_std_target, cur_rate_target = stats.binned_avg_firing_rate_per_neuron(target_spike_train, bin_size=400)


def get_LIF_init_models_stats():
    return main([])


if __name__ == "__main__":
    main(sys.argv[1:])
