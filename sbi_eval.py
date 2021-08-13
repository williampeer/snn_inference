from IO import *
from TargetModels.TargetModels import *
from experiments import poisson_input
from model_util import feed_inputs_sequentially_return_spike_train

from sbi import analysis as analysis
# from sbi import utils as utils
# from sbi.inference.base import infer

t_interval = 6000

experiments_path = '/home/william/repos/archives_snn_inference/archive_multi_sbi_1308/archive/saved/data/'

custom_uuid = 'data'
files_sbi_res = os.listdir(experiments_path + 'sbi_res/')
# files_sbi_samples = os.listdir(experiments_path + 'sbi_samples/')
# experiment_averages = {}
def get_target_observation(model_class, N=3):
    tar_model_fn_lookup = {'LIF_no_grad': lif_continuous_ensembles_model_dales_compliant,
                           'LIF_R_no_grad': lif_r_continuous_ensembles_model_dales_compliant,
                           'LIF_R_ASC_no_grad': lif_r_asc_continuous_ensembles_model_dales_compliant,
                           'GLIF_no_grad': glif_continuous_ensembles_model_dales_compliant}
    tar_in_rate = 10.
    tar_model_fn = tar_model_fn_lookup[model_class.__name__]
    tar_model = tar_model_fn(random_seed=42, N=N)
    inputs = poisson_input(rate=tar_in_rate, t=t_interval, N=N)
    rates = None
    for i in range(10):
        cur_targets = feed_inputs_sequentially_return_spike_train(model=tar_model, inputs=inputs).clone().detach()
        cur_rate = cur_targets.sum(dim=0) * 1000. / cur_targets.shape[0]  # Hz
        if rates is None:
            rates = cur_rate
        else:
            rates = torch.vstack((rates, cur_rate))
    targets = torch.mean(rates, dim=0)
    observation = torch.reshape(targets, (1, -1))

    tar_parameters = tar_model.get_parameters()
    m_name = tar_model.name()

    return observation, tar_parameters, m_name


for sbi_res_file in files_sbi_res:
    print(sbi_res_file)

    full_path = experiments_path + 'sbi_res/'
    full_fname = full_path + sbi_res_file
    print('Loading: {}'.format(full_fname))
    res_load = torch.load(full_fname)
    sut_res = res_load['data']
    sut_description = res_load['dt_descriptor']
    method = 'SNRE'
    print('res load successful.')

    corresponding_samples_fname = 'samples_method_{}_m_name_{}_dt_{}'.format(method, dt_descriptor)
    data_arr = torch.load(full_path + corresponding_samples_fname)['data']
    samples = data_arr['samples']
    observation = data_arr['observation']
    tar_parameters = data_arr['tar_parameters']

    posterior = sut_res[method]
    print(posterior)
    # model_class = sut_res['model_class']
    model_class = GLIF_no_grad
    # N = sut_res['N']
    N = 10

    # observation = get_target_observation(model_class, N)
    # observation, tar_parameters, m_name = get_target_observation(model_class=model_class, N=N)

    # samples = posterior.sample((1000,), x=observation)
    # samples = posterior.sample((10,), x=observation)
    log_probability = posterior.log_prob(samples, x=observation)
    print('log_probability: {}'.format(log_probability))

    try:
        if samples[0].shape[0] <= 3:
            limits_low = torch.zeros((N ** 2 - N,))
            limits_high = torch.ones((N ** 2 - N,))

            for i in range(len(model_class.parameter_names) - 1):
                limits_low = torch.hstack((limits_low, torch.ones((N,)) * model_class.param_lin_constraints[i][0]))
                limits_high = torch.hstack((limits_high, torch.ones((N,)) * model_class.param_lin_constraints[i][1]))

            num_dim = limits_high.shape[0]
            fig, ax = analysis.pairplot(samples, points=tar_parameters,
                                        limits=torch.stack((limits_low, limits_high), dim=1),
                                        figsize=(num_dim, num_dim))
            fig.savefig('./figures/export/analysis_pairplot_{}_one_param_{}_{}.png'.format(method, m_name, sut_description))
    except Exception as e:
        print("except: {}".format(e))


# sut_res = import_data(uuid='sbi_res', fname='')
# sut_samples = import_data(uuid='sbi_samples', fname='')
