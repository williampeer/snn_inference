import sys

import torch

import Constants as C
import data_util
import fit_to_data_exp_suite
from Models.GLIF import GLIF


def main(argv):
    print('Argument List:', str(argv))

    # Default values
    start_seed = 0
    learn_rate = 0.001; N_exp = 5; tau_van_rossum = 4.0; plot_flag = True
    # learn_rate = 0.01; N_exp = 3; tau_van_rossum = 4.0; plot_flag = True

    # max_train_iters = 100; batch_size = 200; rows_per_train_iter = 2000; loss_fn = 'kl_div'
    # max_train_iters = 100; batch_size = 200; rows_per_train_iter = 2000; loss_fn = 'firing_rate_distance'
    # max_train_iters = 50; batch_size = 20; rows_per_train_iter = 4000; loss_fn = 'poisson_nll'
    # max_train_iters = 50; batch_size = 400; rows_per_train_iter = 4000; loss_fn = 'van_rossum_dist'

    # max_train_iters = 100; batch_size = 200; rows_per_train_iter = 2000; loss_fn = 'kldfrd'
    # max_train_iters = 50; batch_size = 20; rows_per_train_iter = 4000; loss_fn = 'pnllfrd'
    max_train_iters = 50; batch_size = 400; rows_per_train_iter = 4000; loss_fn = 'vrdfrd'

    # max_train_iters = 40; batch_size = 200; rows_per_train_iter = 1600; loss_fn = 'mse'

    optimiser = 'Adam'
    initial_poisson_rate = 0.6  # per ms

    evaluate_step = 1
    # evaluate_step = int(max(max_train_iters/10, 1))
    # data_path = None
    # prefix = '/Users/william/data/target_data/'
    target_data_folder_path = data_util.prefix + data_util.path

    # tmn = 0
    trn = 0
    output_fnames_rate_0_6 = ['generated_spike_train_random_glif_1_model_t_300s_rate_0_6.mat',
                              'generated_spike_train_random_glif_2_model_t_300s_rate_0_6.mat',
                              'generated_spike_train_random_glif_3_model_t_300s_rate_0_6.mat',
                              'generated_spike_train_glif_slower_rate_async_t_300s_rate_0_6.mat',
                              'generated_spike_train_glif_slower_more_synchronous_model_t_300s_rate_0_6.mat']
    output_fnames_rate_0_4 = []
    target_params_rate_0_6 = []
    target_params_rate_0_4 = []
    for fn in output_fnames_rate_0_6:
        output_fnames_rate_0_4.append(fn.replace('_6', '_4'))
        target_params_rate_0_6.append(fn.replace('.mat', '_params.pt'))
        target_params_rate_0_4.append(fn.replace('_6.mat', '_4_params.pt'))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]
    for i, opt in enumerate(opts):
        if opt == '-h':
            print('main.py -s <script> -lr <learning-rate> -ti <training-iterations> -N <number-of-experiments> '
                  '-bs <batch-size> -tvr <van-rossum-time-constant> -rpti <rows-per-training-iteration> '
                  '-optim <optimiser> -ipr <initial-poisson-rate> -es <evaluate-step> -tmn <target-model-number> '
                  '-trn <target-rate-number>')
            sys.exit()
        elif opt in ("-lr", "--learning-rate"):
            learn_rate = float(args[i])
        elif opt in ("-ti", "--training-iterations"):
            max_train_iters = int(args[i])
        elif opt in ("-N", "--numbers-of-experiments"):
            N_exp = int(args[i])
        elif opt in ("-bs", "--batch-size"):
            batch_size = int(args[i])
        elif opt in ("-tvr", "--van-rossum-time-constant"):
            tau_van_rossum = float(args[i])
        elif opt in ("-rpti", "--rows-per-training-iteration"):
            rows_per_train_iter = int(args[i])
        elif opt in ("-optim", "--optimiser"):
            optimiser = str(args[i])
        elif opt in ("-ipr", "--initial-poisson-rate"):
            initial_poisson_rate = float(args[i])
        elif opt in ("-es", "--evaluate-step"):
            evaluate_step = int(args[i])
        elif opt in ("-dp", "--data-path"):
            data_path = args[i]
        elif opt in ("-sp", "--should-plot"):
            plot_flag = bool(args[i])
        # elif opt in ("-tmn", "--target-model-number"):
        #     tmn = int(args[i])
        elif opt in ("-trn", "--target-rate-number"):
            trn = int(args[i])
        elif opt in ("-ss", "--start-seed"):
            start_seed = int(args[i])

    if trn == 0:
        output_fnames = output_fnames_rate_0_6
        target_fnames = target_params_rate_0_6
    elif trn == 1:
        output_fnames = output_fnames_rate_0_4
        target_fnames = target_params_rate_0_4
    else:
        raise NotImplementedError()

    for f_i in range(len(output_fnames)):
        data_path = target_data_folder_path + output_fnames[f_i]
        target_params_dict = torch.load(target_data_folder_path + target_fnames[f_i])
        target_parameters = {}
        for param_i, param in enumerate(target_params_dict.values()):
            target_parameters[param_i-1] = param.clone().detach().numpy()

        constants = C.Constants(learn_rate=learn_rate, train_iters=max_train_iters, N_exp=N_exp, batch_size=batch_size,
                                tau_van_rossum=tau_van_rossum, rows_per_train_iter=rows_per_train_iter, optimiser=optimiser,
                                initial_poisson_rate=initial_poisson_rate, loss_fn=loss_fn, evaluate_step=evaluate_step,
                                data_path=data_path, plot_flag=plot_flag, start_seed=start_seed, target_fname=target_fnames[f_i])

        # models = [LIF, LIF_R, LIF_ASC, LIF_R_ASC, GLIF]
        models = [GLIF]
        for m_class in models:
            fit_to_data_exp_suite.start_exp(constants=constants, model_class=m_class, target_parameters=target_parameters)


if __name__ == "__main__":
    main(sys.argv[1:])
