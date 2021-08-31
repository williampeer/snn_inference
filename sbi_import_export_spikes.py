from IO import *
from TargetModels.TargetModels import *

from spike_train_matlab_export import simulate_and_save_model_spike_train


def convert_posterior_to_model_params_dict(model_class, posterior_params, N):
    if posterior_params.shape[0] == 1:
        posterior_params = posterior_params[0]
    model_params = {}
    p_i = 0
    for p_name in model_class.parameter_names:
        if p_i == 0:
            assert p_name == 'w'
            model_params[p_name] = posterior_params[:(N**2-N)]
        else:
            model_params[p_name] = posterior_params[(N**2-N)+N*(p_i-1):(N**2-N)+N*p_i]
        p_i += 1
    return model_params


def main():
    # experiments_path = '/home/william/repos/archives_snn_inference/archive_multi_sbi_1308/archive/saved/data/'
    # experiments_path = '/home/william/repos/archives_snn_inference/archive_sbi_runs_1608/archive/saved/data/'  # Single

    # experiments_path = '/home/william/repos/archives_snn_inference/archive_1208_GLIF_3_LIF_R_AND_ASC_10_PLUSPLUS/archive/saved/data/'  # Done (export)
    # experiments_path = '/home/william/repos/archives_snn_inference/archive_sbi_runs_1608/archive/saved/data/'
    # experiments_path = '/home/william/repos/archives_snn_inference/archive_1908_multi_N_3_10/archive/saved/data/'
    experiments_path = '/home/william/repos/archives_snn_inference/archive_3008_all_seed_64_and_sbi_3_and_4/archive/saved/data/'

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
        tar_seed = False
        if sbi_res_file.__contains__('tar_seed'):
            tar_seed = int(sbi_res_file.split('tar_seed_')[-1].split('.pt')[0])

        if 'param_num' in sbi_res:
            param_num = sbi_res['param_num']
            if tar_seed:
                corresponding_samples_fname = 'samples_method_{}_m_name_{}_param_num_{}_dt_{}_tar_seed_{}.pt'.format(method, m_name, param_num, dt_descriptor, tar_seed)
            else:
                corresponding_samples_fname = 'samples_method_{}_m_name_{}_param_num_{}_dt_{}.pt'.format(method, m_name, param_num, dt_descriptor)

            print('single param. passing for now..')
        else:
            if tar_seed:
                corresponding_samples_fname = 'samples_method_{}_m_name_{}_dt_{}_tar_seed_{}.pt'.format(method, m_name, dt_descriptor, tar_seed)
            else:
                corresponding_samples_fname = 'samples_method_{}_m_name_{}_dt_{}.pt'.format(method, m_name, dt_descriptor)

            try:
                print('sbi_res load successful.')
                data_arr = torch.load(experiments_path + 'sbi_samples/' + corresponding_samples_fname)['data']
                print('sbi_samples load successful.')

                save_fname = 'export_{}_tar_seed_{}_sample_N_{}'.format(corresponding_samples_fname.strip('.pt')+'', tar_seed, N)

                torch.manual_seed(tar_seed)
                np.random.seed(tar_seed)

                if not os.path.exists(data_util.prefix + data_util.path + save_fname + '.mat'):
                    # samples = data_arr['samples']
                    observation = data_arr['observation']
                    # points = data_arr['tar_parameters']
                    m_name = data_arr['m_name']

                    # log_probability = posterior.log_prob(samples, x=observation)
                    # print('log_probability: {}'.format(log_probability))

                    print('drawing most likely sample..')
                    N_samples = 20
                    posterior_params = posterior.sample((N_samples,), x=observation)
                    print('\nposterior_params: {}'.format(posterior_params))

                    for s_i in range(N_samples):
                        model_params = convert_posterior_to_model_params_dict(model_class, posterior_params[s_i], N)
                        programmatic_neuron_types = torch.ones((N,))
                        for n_i in range(int(2 * N / 3), N):
                            programmatic_neuron_types[n_i] = -1
                        model = model_class(parameters=model_params, N=N, neuron_types=programmatic_neuron_types)

                        # makedir_if_not_exists('./figures/default/plot_imported_model/' + archive_name)
                        # load_and_export_sim_data(full_folder_path + f, fname=archive_name + cur_fname)
                        simulate_and_save_model_spike_train(model, 10., 60*1000, None, m_name, fname=save_fname+'_sample_N_{}'.format(s_i))
                else:
                    print('file exists. skipping..')

            except Exception as e:
                print('😱😱😱')
                print(e)


# sut_res = import_data(uuid='sbi_res', fname='')
# sut_samples = import_data(uuid='sbi_samples', fname='')

if __name__ == "__main__":
    main()
    # sys.exit(0)

