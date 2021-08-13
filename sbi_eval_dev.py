from sbi import analysis as analysis

from IO import *
from TargetModels.TargetModels import *

# from sbi import utils as utils
# from sbi.inference.base import infer

# t_interval = 6000

# experiments_path = '/home/william/repos/archives_snn_inference/archive_multi_sbi_1308/archive/saved/data/'
experiments_path = '/home/william/repos/snn_inference/saved/data/'

custom_uuid = 'data'
files_sbi_res = os.listdir(experiments_path + 'sbi_res/')

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
    else:
        corresponding_samples_fname = 'samples_method_{}_m_name_{}_dt_{}.pt'.format(method, m_name, dt_descriptor)
    print('sbi_res load successful.')

    try:
        data_arr = torch.load(experiments_path + 'sbi_samples/' + corresponding_samples_fname)['data']
        print('sbi_samples load successful.')
        samples = data_arr['samples']
        observation = data_arr['observation']
        points = data_arr['tar_parameters']
        m_name = data_arr['m_name']

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
                fig, ax = analysis.pairplot(samples, points=points,
                                            limits=torch.stack((limits_low, limits_high), dim=1),
                                            figsize=(num_dim, num_dim))
                fig.savefig('./figures/export/analysis_pairplot_{}_one_param_{}_{}.png'.format(method, m_name, sut_description))
        except Exception as e:
            print("plotting failed. exception: {}".format(e))
    except Exception as e:
        print("loading corresponding samples-file failed. exception: {}".format(e))


# sut_res = import_data(uuid='sbi_res', fname='')
# sut_samples = import_data(uuid='sbi_samples', fname='')
