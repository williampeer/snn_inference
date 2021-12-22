import sys

import numpy as np

import model_util
import plot
from IO import *
from Models.microGIF import microGIF
from analysis import analysis_util
from analysis.spike_train_matlab_export import simulate_and_save_model_spike_train


def OU_process_input(t=4000):
    mu = np.array([1., 1., 0, 1.])
    tau = np.array([2., 2., 1., 2.])
    q = np.array([0.5, 0.5, 0, 0.5])
    dW = np.random.normal()

    I_0 = np.array([0.5, 0.5, 0, 0.5])
    I_interval = np.asarray(I_0)
    I = np.asarray(I_0)
    for t_i in range(t):
        dI = (I - mu)/tau + np.sqrt(2/tau) * q * dW
        I = I + dI
        I_interval = np.concatenate((I_interval, dI))

    assert I_interval.shape[0] == t and I_interval.shape[1] == 4, "I_interval should be {}x{}. Was: {}".format(t, 4, I_interval.shape)
    return I_interval


def get_model_activity(model, inputs):
    if model.__class__ is microGIF:
        s_lambdas, spikes, _ = model_util.feed_inputs_sequentially_return_args(model=model, inputs=inputs.clone().detach())
    else:
        s_lambdas, spikes = model_util.feed_inputs_sequentially_return_tuple(model=model, inputs=inputs.clone().detach())
    return s_lambdas, spikes


def spike_correlations(m_spikes, t_spikes):

    return 0.


def activity_RMSE(m_act, t_act):

    return 0.


def convert_posterior_to_model_params_dict(model_class, posterior_params, target_class, target_points, N):
    if posterior_params.shape[0] == 1:
        posterior_params = posterior_params[0]
    model_params = {}
    p_i = 0
    for p_name in model_class.free_parameters:
        if p_i == 0:
            assert p_name == 'w'
            model_params[p_name] = posterior_params[:(N**2-N)]
        else:
            model_params[p_name] = posterior_params[(N**2-N)+N*(p_i-1):(N**2-N)+N*p_i]
        p_i += 1

    t_p_i = 1
    for t_p_name in target_class.free_parameters:
        if t_p_name not in model_params:
            model_params[t_p_name] = target_points[(N**2-N)+N*(t_p_i-1):(N**2-N)+N*t_p_i]
        t_p_i += 1

    return model_params


# experiments_path = '/home/william/repos/archives_snn_inference/GENERIC/archive/saved/data/'
experiments_path = '/home/william/repos/archives_snn_inference/archive_1612/archive/saved/data/'

files_sbi_res = os.listdir(experiments_path + 'sbi_res/')

correlations_wn_per_model_type = { 'LIF': [], 'GLIF': [], 'microGIF': [] }
correlations_OU_per_model_type = { 'LIF': [], 'GLIF': [], 'microGIF': [] }
activity_rmse_wn_per_model_type = { 'LIF': [], 'GLIF': [], 'microGIF': [] }
activity_rmse_OU_per_model_type = { 'LIF': [], 'GLIF': [], 'microGIF': [] }
for sbi_res_file in files_sbi_res:
    print(sbi_res_file)

    sbi_res_path = experiments_path + 'sbi_res/' + sbi_res_file
    print('Loading: {}'.format(sbi_res_path))
    res_load = torch.load(sbi_res_path)
    print('sbi_res load successful.')

    sbi_res = res_load['data']
    assert sbi_res.keys().__contains__('SNPE'), "method SNPE expected"
    method = 'SNPE'
    posterior = sbi_res[method]
    sut_description = sbi_res['dt_descriptor']
    model_class = sbi_res['model_class']
    m_name = model_class.__name__
    N = sbi_res['N']
    dt_descriptor = sbi_res['dt_descriptor']
    tar_seed = False
    if sbi_res_file.__contains__('tar_seed'):
        tar_seed = int(sbi_res_file.split('tar_seed_')[-1].split('.pt')[0])

    if tar_seed:
        corresponding_samples_fname = 'samples_method_{}_m_name_{}_dt_{}_tar_seed_{}.pt'.format(method, m_name, dt_descriptor, tar_seed)
    else:
        corresponding_samples_fname = 'samples_method_{}_m_name_{}_dt_{}.pt'.format(method, m_name, dt_descriptor)

    data_arr = torch.load(experiments_path + 'sbi_samples/' + corresponding_samples_fname)['data']
    print('sbi_samples load successful.')

    save_fname = 'spikes_sbi_{}_tar_seed_{}'.format(corresponding_samples_fname.strip('.pt')+'', tar_seed)

    torch.manual_seed(tar_seed)
    np.random.seed(tar_seed)

    # samples = data_arr['samples']
    observation = data_arr['observation']
    # points = data_arr['tar_parameters']
    m_name = data_arr['m_name']

    # log_probability = posterior.log_prob(samples, x=observation)
    # print('log_probability: {}'.format(log_probability))

    N_samples = 10
    print('Drawing the {} most likely samples..'.format(N_samples))
    posterior_params = posterior.sample((N_samples,), x=observation)
    print('\nposterior_params: {}'.format(posterior_params))
    tar_m_name = m_name
    if m_name == 'microGIF':
        tar_m_name = 'mesoGIF'
    target_model = analysis_util.get_target_model(tar_m_name)

    for s_i in range(N_samples):
        model_params = convert_posterior_to_model_params_dict(model_class, posterior_params[s_i], target_class=model_class, target_points=[], N=N)
        programmatic_neuron_types = torch.ones((N,))
        for n_i in range(int(2 * N / 3), N):
            programmatic_neuron_types[n_i] = -1
        if model_class is microGIF:
            model = model_class(parameters=model_params, N=N)
        else:
            model = model_class(parameters=model_params, N=N, neuron_types=programmatic_neuron_types)

        # burn in:
        burn_in_inputs = OU_process_input(t=9000)
        _, _ = get_model_activity(model, burn_in_inputs)
        _, _ = get_model_activity(target_model, burn_in_inputs)
        eval_inputs = OU_process_input(t=9000)
        m_act, m_spikes = get_model_activity(model, burn_in_inputs)
        t_act, t_spikes = get_model_activity(target_model, burn_in_inputs)

        # spike_correlation_white_noise = 0.
        # activity_RMSE_white_noise = 0.
        spike_correlation_OU_process = spike_correlations(m_spikes, t_spikes)
        activity_RMSE_OU_process = activity_RMSE(m_act, t_act)

xticks = []; correlations_wn = []; rmse_wn = []; correlations_OU = []; rmse_OU = []
for m_k in correlations_wn_per_model_type.keys():
    # rates.append(np.mean(rates_per_model_type[m_k]))
    # rate_stds.append(np.std(rates_per_model_type[m_k]))
    # dists.append(np.mean(p_dists_per_model_type[m_k]))
    # dist_stds.append(np.std(p_dists_per_model_type[m_k]))
    # init_dists.append(np.mean(init_dists_per_model_type[m_k]))
    # init_dist_stds.append(np.std(init_dists_per_model_type[m_k]))
    xticks.append(m_k.replace('microGIF', 'miGIF').replace('mesoGIF', 'meGIF'))

plot_exp_type = 'export_sbi'
# plot.bar_plot_neuron_rates(0, correlations_wn, 0, correlations_wn_std,
#                            exp_type=plot_exp_type, uuid='all', fname='sbi_mean_p_dist_all.eps', xticks=xticks,
#                            custom_legend=['None', 'Fitted models'], ylabel='Avg. param dist.')
# plot.bar_plot_neuron_rates(0, rmse_wn, 0., rmse_wn_std, plot_exp_type, 'all',
#                            custom_legend=['Target models', 'Fitted models'],
#                            fname='plot_rates_all.eps', xticks=xticks,
#                            custom_colors=['Green', 'Magenta'])

# sys.exit()
