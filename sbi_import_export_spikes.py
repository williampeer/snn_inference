from sbi import analysis as analysis

import plot
from IO import *
from TargetModels.TargetModels import *

# from sbi import utils as utils
# from sbi.inference.base import infer

# t_interval = 6000

# experiments_path = '/home/william/repos/archives_snn_inference/archive_multi_sbi_1308/archive/saved/data/'
# experiments_path = '/home/william/repos/archives_snn_inference/archive_sbi_runs_1608/archive/saved/data/'  # Single
from experiments import generate_synthetic_data
from spike_train_matlab_export import load_and_export_sim_data, simulate_and_save_model_spike_train

experiments_path = '/home/william/repos/archives_snn_inference/archive_1208_GLIF_3_LIF_R_AND_ASC_10_PLUSPLUS/archive/saved/data/'
# experiments_path = '/home/william/repos/snn_inference/saved/data/'

custom_uuid = 'data'
files_sbi_res = os.listdir(experiments_path + 'sbi_res/')


def export_plots(samples, points, lim_low, lim_high, description):
    num_dim = lim_high.shape[0]
    if num_dim < 12:  # full marginal plot
        fig, ax = analysis.pairplot(samples, points=points, limits=torch.stack((lim_low, lim_high)), figsize=(num_dim, num_dim))
        fig.savefig('./figures/export_analysis_pairplot_{}_one_param_{}_{}.png'.format(method, m_name, description))
    else:
        # TODO: calc avg, plot mean:
        # TODO: plot sub-set
        samples
        # fig_subset, ax = analysis.pairplot(samples, points=points, limits=torch.stack((lim_low, lim_high)), figsize=(10, 10))
        # fig_subset.savefig('./figures/export_subset_analysis_pairplot_{}_one_param_{}_{}.png'.format(method, m_name, description))
        pass


def limits_for_class(model_class, N):
    # parsed_weights = torch.zeros((N ** 2 - N,))
    limits_low = torch.zeros((N ** 2 - N,))
    limits_high = torch.ones((N ** 2 - N,))

    for i in range(len(model_class.parameter_names) - 1):
        limits_low = torch.hstack((limits_low, torch.ones((N,)) * model_class.param_lin_constraints[i][0]))
        limits_high = torch.hstack((limits_high, torch.ones((N,)) * model_class.param_lin_constraints[i][1]))

    return limits_low, limits_high


def export_stats_model_target(model, observation, descriptor):
    spike_rates = torch.mean(generate_synthetic_data(model, poisson_rate=10., t=6000.), dim=0)
    for spike_iters in range(10-1):
        spike_rates = torch.cat([spike_rates, torch.mean(generate_synthetic_data(model, poisson_rate=10., t=6000.), dim=0)])
    mean_spike_rates = torch.mean(spike_rates, dim=0)

    plot.bar_plot_pair_custom_labels_two_grps(y1=mean_spike_rates, y2=torch.mean(observation, dim=0),
                                              y1_std=torch.std(spike_rates, dim=0), y2_std=torch.std(observation, dim=0),
                                              labels=['Neuron', 'Firing rate ($Hz$)'],
                                              exp_type='export', uuid='ho_stats' + '/' + custom_uuid,
                                              fname='export_bar_plot_avg_rate_sbi_{}.eps'.format(descriptor),
                                              title='Avg. rates for SBI parameters ({})'.format(descriptor),
                                              ylabel='Firing rate ($Hz$)',
                                              legend=['Fitted', 'Target'])


def convert_posterior_to_model_params_dict(model_class, posterior_params):
    model_params = {}
    p_i = 0
    for p_name in model_class.parameter_names:
        model_params[p_name] = posterior_params[p_i]
        p_i += 1


for sbi_res_file in files_sbi_res:
    print(sbi_res_file)

    sbi_res_path = experiments_path + 'sbi_res/' + sbi_res_file
    print('Loading: {}'.format(sbi_res_path))
    res_load = torch.load(sbi_res_path)
    sbi_res = res_load['data']
    sut_description = sbi_res['dt_descriptor']
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

        export_plots(samples, points, lim_low, lim_high, dt_descriptor)

        print('drawing most likely sample..')
        posterior_params = posterior.sample((1,), x=observation)
        print('\nposterior_params: {}'.format(posterior_params))

        model_params = convert_posterior_to_model_params_dict(model_class, posterior_params)
        model = model_class(parameters=posterior_params)
        simulate_and_save_model_spike_train(model, 10., 60*1000, None, m_name, fname='export_{}'.format(corresponding_samples_fname))
        export_stats_model_target(model, observation=observation, descriptor='{}_parallel_sbi'.format(model.name()))

        # except Exception as e:
        #     print('😱😱😱')
        #     print(e)


# sut_res = import_data(uuid='sbi_res', fname='')
# sut_samples = import_data(uuid='sbi_samples', fname='')
