import os
import matplotlib.pyplot as plt

from sbi import analysis as analysis

import parameter_distance
import plot
from TargetModels.TargetModels import *
from experiments import generate_synthetic_data
from sbi_import_export_spikes import convert_posterior_to_model_params_dict


def export_plots(samples, points, lim_low, lim_high, N, method, m_name, description):
    num_dim = lim_high.shape[0]
    if num_dim < 12:  # full marginal plot
        plt.figure()
        fig, ax = analysis.pairplot(samples, points=points, limits=torch.stack((lim_low, lim_high)), figsize=(num_dim, num_dim))
        fig.savefig('./figures/export_analysis_pairplot_{}_one_param_{}_{}.png'.format(method, m_name, description))
        plt.close()
    else:
        plt.figure()
        weights_offset = N ** 2 - N
        sample_means = [torch.mean(samples[:, :weights_offset])]
        lim_low_means = [torch.mean(lim_low[:weights_offset])]
        lim_high_means = [torch.mean(lim_high[:weights_offset])]
        pt_means = [torch.mean(points[0])]
        for p_j in range(1, len(points)):
            sample_means.append(torch.mean(samples[:, weights_offset+(p_j-1)*N:weights_offset+p_j*N]))
            lim_low_means.append(torch.mean(lim_low[weights_offset+(p_j-1)*N:weights_offset+p_j*N]))
            lim_high_means.append(torch.mean(lim_high[weights_offset+(p_j-1)*N:weights_offset+p_j*N]))
            pt_means.append(torch.mean(points[p_j]))
        fig_subset_mean, ax_mean = analysis.pairplot(torch.tensor([sample_means]).T, points=torch.tensor([pt_means]).T,
                                                     limits=torch.stack((torch.tensor([lim_low_means]).T, torch.tensor([lim_high_means]).T)),
                                                     figsize=(num_dim, num_dim))
        fig_subset_mean.savefig('./figures/export_sut_means_analysis_pairplot_{}_one_param_{}_{}.png'.format(method, m_name, description))
        plt.close()

        # Marginals only for p_i, p_i
        for p_i in range(1, len(points)):
            plt.figure()
            fig_subset_mean, ax_mean = analysis.pairplot(samples[:, weights_offset+(p_i-1)*N:weights_offset+p_i*N],
                                                         points=points[p_i],
                                                         limits=torch.reshape(torch.stack((lim_low[weights_offset+(p_i-1)*N:weights_offset+p_i*N],
                                                                                           lim_high[weights_offset+(p_i-1)*N:weights_offset+p_i*N])), (N, 2)),
                                                         figsize=(N, N))
            fig_subset_mean.savefig('./figures/export_sut_subset_analysis_pairplot_{}_{}_one_param_{}_{}.png'.format(method, m_name, p_i, description))
            plt.close()
        # pass


def export_stats_model_target(model, observation, descriptor):
    spike_train, _ = generate_synthetic_data(model, poisson_rate=10., t=6000)
    spike_rates = 1000. * spike_train.sum(dim=0) / spike_train.shape[0]
    for spike_iters in range(10-1):
        spike_train, _ = generate_synthetic_data(model, poisson_rate=10., t=6000)
        spike_rates = torch.cat([spike_rates, 1000. * spike_train.sum(dim=0) / spike_train.shape[0]])

    spike_rates = torch.reshape(spike_rates, (-1, model.N))
    mean_spike_rates = torch.mean(spike_rates, dim=0)
    rate_stds = torch.std(spike_rates, dim=0)
    custom_uuid = 'sbi'
    plt.figure()
    plot.bar_plot_pair_custom_labels(y1=mean_spike_rates, y2=observation[0],
                                     y1_std=rate_stds, y2_std=torch.zeros_like(rate_stds),
                                     labels=range(spike_train.shape[1]),
                                     exp_type='export', uuid='ho_stats' + '/' + custom_uuid,
                                     fname='export_bar_plot_avg_rate_sbi_{}.eps'.format(descriptor),
                                     title='Avg. rates for SBI parameters ({})'.format(descriptor),
                                     ylabel='Firing rate ($Hz$)', xlabel='Neuron',
                                     legend=['Fitted', 'Target'])
    plt.close()

    # correlation:
    # Not too informative, unless the same input is used. However, correlation between neurons within model may be informative about that model, but so is NMF.

    return mean_spike_rates


def export_stats_top_samples(mean_model_rates, std_model_rates, targets, descriptor):
    plt.figure()
    plot.bar_plot_pair_custom_labels(y1=mean_model_rates, y2=targets,
                                     y1_std=std_model_rates, y2_std=torch.zeros_like(std_model_rates),
                                     labels=range(len(mean_model_rates)),
                                     exp_type='export', uuid='ho_stats' + '/' + 'sbi',
                                     fname='export_bar_plot_avg_rate_sbi_{}.eps'.format(descriptor),
                                     title='Avg. rates {} most likely samples ({})'.format(len(mean_model_rates), descriptor),
                                     ylabel='Firing rate ($Hz$)', xlabel='Neuron',
                                     legend=['Fitted', 'Target'])
    plt.close()


def limits_for_class(model_class, N):
    # parsed_weights = torch.zeros((N ** 2 - N,))
    limits_low = torch.zeros((N ** 2 - N,))
    limits_high = torch.ones((N ** 2 - N,))

    for i in range(1, len(model_class.parameter_names)):
        limits_low = torch.hstack((limits_low, torch.ones((N,)) * model_class.param_lin_constraints[i][0]))
        limits_high = torch.hstack((limits_high, torch.ones((N,)) * model_class.param_lin_constraints[i][1]))

    return limits_low, limits_high


def plot_param_dist(parameter_distance, title, fname):
    plot.bar_plot(parameter_distance, parameter_distance, False, 'export',
                  'sbi_param_dist', 'export_sbi_param_dist_{}.png'.format(fname), title)


def main():
    experiments_path = '/home/william/repos/archives_snn_inference/archive_1208_GLIF_3_LIF_R_AND_ASC_10_PLUSPLUS/archive/saved/data/'
    # experiments_path = '/home/william/repos/archives_snn_inference/archive_1908_multi_N_3_10/archive/saved/data/'
    # experiments_path = '/home/william/repos/snn_inference/saved/data/'

    custom_uuid = 'data'
    files_sbi_res = os.listdir(experiments_path + 'sbi_res/')

    for sbi_res_file in files_sbi_res:
        print(sbi_res_file)

        sbi_res_path = experiments_path + 'sbi_res/' + sbi_res_file
        print('Loading: {}'.format(sbi_res_path))
        res_load = torch.load(sbi_res_path)
        sbi_res = res_load['data']
        method = 'SNRE'
        posterior = sbi_res[method]
        model_class = sbi_res['model_class']
        m_name = model_class.__name__.strip('_no_grad')
        N = sbi_res['N']
        dt_descriptor = sbi_res['dt_descriptor']
        if 'param_num' in sbi_res:
            param_num = sbi_res['param_num']
            corresponding_samples_fname = 'samples_method_{}_m_name_{}_param_num_{}_dt_{}.pt'.format(method, m_name, param_num, dt_descriptor)

            print('single param. passing for now..')
        else:
            corresponding_samples_fname = 'samples_method_{}_m_name_{}_dt_{}.pt'.format(method, m_name, dt_descriptor)
            print('sbi_res load successful.')

            # try:
            data_arr = torch.load(experiments_path + 'sbi_samples/' + corresponding_samples_fname)['data']
            print('sbi_samples load successful.')
            samples = data_arr['samples']
            observation = data_arr['observation']
            points = data_arr['tar_parameters']
            m_name = data_arr['m_name']

            lim_low, lim_high = limits_for_class(model_class, N=N)

            # log_probability = posterior.log_prob(samples, x=observation)
            # print('log_probability: {}'.format(log_probability))

            export_plots(samples, points, lim_low, lim_high, N, method, m_name, dt_descriptor)

            print('drawing most likely sample..')
            N_samples = 20
            posterior_params = posterior.sample((N_samples,), x=observation)
            print('\nposterior_params: {}'.format(posterior_params))

            mean_model_rates = torch.tensor([])
            # std_model_rates = torch.tensor([])

            avg_param_dist_across_samples = []
            for s_i in range(N_samples):
                model_params = convert_posterior_to_model_params_dict(model_class, posterior_params[s_i], N)
                programmatic_neuron_types = torch.ones((N,))
                for n_i in range(int(2 * N / 3), N):
                    programmatic_neuron_types[n_i] = -1
                model = model_class(parameters=model_params, N=N, neuron_types=programmatic_neuron_types)
                cur_mean_model_rates = export_stats_model_target(model, observation=observation,
                                                                 descriptor='{}_parallel_sbi_{}_sample_N_{}'.
                                                                    format(m_name, dt_descriptor, s_i))
                mean_model_rates = torch.cat((mean_model_rates, cur_mean_model_rates))

                current_avg_dist_per_p = []
                model_parameter_list = model.get_parameters()
                for p_i in range(len(model_parameter_list)):
                    dist_p_i = parameter_distance.euclid_dist(model_parameter_list[p_i], points[p_i])
                    current_avg_dist_per_p.append(dist_p_i)
                plot_param_dist(np.array(current_avg_dist_per_p), 'Parameter distance for sample: {}'.format(s_i),
                                '{}_N_{}_parallel_sbi_{}_sample_num_{}'.format(m_name, N, dt_descriptor, s_i))
                avg_param_dist_across_samples.append(current_avg_dist_per_p)
            plot_param_dist(np.mean(avg_param_dist_across_samples, axis=0), 'Parameter distance across samples',
                            'sbi_samples_avg_param_dist_{}_N_{}_{}'.format(m_name, N, dt_descriptor))

                # std_model_rates.append(cur_std_model_rate)
            mean_model_rates = torch.reshape(mean_model_rates, (N_samples, -1))
            export_stats_top_samples(torch.mean(mean_model_rates, dim=0), torch.std(mean_model_rates, dim=0),
                                     observation[0], '{}_{}_sbi_parallel_{}'.format(method, m_name, dt_descriptor))

if __name__ == "__main__":
    main()
    # sys.exit(0)

