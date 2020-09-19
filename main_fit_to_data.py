import sys

import torch

import Constants as C
import data_util
from Models.GLIF import GLIF


def main(argv):
    print('Argument List:', str(argv))

    # Default values
    learn_rate = 0.005; N_exp = 5; tau_van_rossum = 4.0

    # max_train_iters = 40; batch_size = 200; rows_per_train_iter = 1600; loss_fn = 'kl_div'
    # max_train_iters = 200; batch_size = 10; rows_per_train_iter = 400; loss_fn = 'poisson_nll'
    # max_train_iters = 100; batch_size = 10; rows_per_train_iter = 100; loss_fn = 'poisson_nll'
    max_train_iters = 100; batch_size = 400; rows_per_train_iter = 2000; loss_fn = 'van_rossum_dist'

    optimiser = 'Adam'
    initial_poisson_rate = 0.6

    evaluate_step = 1
    # data_path = None
    # prefix = '/Users/william/data/target_data/'
    target_data_path = data_util.prefix + data_util.path
    data_path = target_data_path + 'generated_spike_train_random_glif_model_t_300s_rate_0_6.mat'
    target_params_dict = torch.load(target_data_path + 'generated_spike_train_random_glif_model_t_300s_rate_0_6_params.pt')
    target_parameters = {}
    for param_i, param in enumerate(target_params_dict.values()):
        target_parameters[param_i] = [param.clone().detach().numpy()]

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]

    for i, opt in enumerate(opts):
        if opt == '-h':
            print('main.py -s <script> -lr <learning-rate> -ti <training-iterations> -N <number-of-experiments> '
                  '-bs <batch-size> -tvr <van-rossum-time-constant> -rpti <rows-per-training-iteration> '
                  '-optim <optimiser> -ipr <initial-poisson-rate> -es <evaluate-step>')
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

    constants = C.Constants(learn_rate=learn_rate, train_iters=max_train_iters, N_exp=N_exp, batch_size=batch_size,
                            tau_van_rossum=tau_van_rossum, rows_per_train_iter=rows_per_train_iter, optimiser=optimiser,
                            initial_poisson_rate=initial_poisson_rate, loss_fn=loss_fn, evaluate_step=evaluate_step,
                            data_path=data_path)

    import fit_to_data_exp_suite
    # models = [LIF, LIF_R, LIF_ASC, LIF_R_ASC, GLIF]
    models = [GLIF]
    for m_class in models:
        fit_to_data_exp_suite.start_exp(constants=constants, model_class=m_class, target_parameters=target_parameters)


if __name__ == "__main__":
    main(sys.argv[1:])
