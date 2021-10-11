import sys

import Constants as C
import gif_exp_suite
from Models.microGIF import microGIF
from TargetModels import TargetModelMicroGIF
from eval import LossFn


def main(argv):
    print('Argument List:', str(argv))

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print('Using {} device'.format(device))

    # Default values
    start_seed = 42
    # exp_type_str = C.ExperimentType.SanityCheck.name
    exp_type_str = C.ExperimentType.Synthetic.name
    # exp_type_str = C.ExperimentType.DataDriven.name
    learn_rate = 0.015; N_exp = 3; tau_van_rossum = 20.0; plot_flag = True
    # Run 100 with lr 0.01 and 0.02
    max_train_iters = 100
    num_targets = 1
    # Q: Interval size effect on loss curve and param retrieval for both lfns
    interval_size = 8000
    batch_size = interval_size; rows_per_train_iter = interval_size
    bin_size = int(interval_size/10)  # for RPH
    # burn_in = False
    burn_in = True

    optimiser = 'SGD'
    # network_size = 2
    # network_size = 4
    network_size = 8
    # network_size = 16

    evaluate_step = 1
    data_path = None

    model_type = 'microGIF'
    loss_fn = 'frd'
    # loss_fn = 'vrd'
    norm_grad_flag = False

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]
    for i, opt in enumerate(opts):
        if opt == '-h':
            print('main.py -s <script> -lr <learning-rate> -ti <training-iterations> -N <number-of-experiments> '
                  '-bs <batch-size> -tvr <van-rossum-time-constant> -rpti <rows-per-training-iteration> '
                  '-optim <optimiser> -ipr <initial-poisson-rate> -es <evaluate-step> -tmn <target-model-number> '
                  '-ss <start-seed> -et <experiment-type> -mt <model-type> -spf <silent-penalty-factor> '
                  '-ng <normalised-gradients> -dp <data-path>')
            sys.exit()
        elif opt in ("-lr", "--learning-rate"):
            learn_rate = float(args[i])
        elif opt in ("-ti", "--training-iterations"):
            max_train_iters = int(args[i])
        elif opt in ("-noe", "--numbers-of-experiments"):
            N_exp = int(args[i])
        elif opt in ("-bas", "--batch-size"):
            batch_size = int(args[i])
        elif opt in ("-bis", "--bin-size"):
            bin_size = int(args[i])
        elif opt in ("-tvr", "--van-rossum-time-constant"):
            tau_van_rossum = float(args[i])
        elif opt in ("-rpti", "--rows-per-training-iteration"):
            rows_per_train_iter = int(args[i])
        elif opt in ("-o", "--optimiser"):
            optimiser = str(args[i])
        elif opt in ("-es", "--evaluate-step"):
            evaluate_step = int(args[i])
        elif opt in ("-lfn", "--loss-function"):
            loss_fn = args[i]
        elif opt in ("-sp", "--should-plot"):
            plot_flag = bool(args[i])
        elif opt in ("-ss", "--start-seed"):
            start_seed = int(args[i])
        elif opt in ("-et", "--experiment-type"):
            exp_type_str = args[i]
        elif opt in ("-mt", "--model-type"):
            model_type = args[i]
        elif opt in ("-ng", "--normalised-gradients"):
            norm_grad_flag = bool(args[i])
        elif opt in ("-dp", "--data-path"):
            data_path = str(args[i])
        elif opt in ("-N", "--network-size"):
            network_size = int(args[i])
        elif opt in ("-nt", "--num-targets"):
            num_targets = int(args[i])
            assert num_targets > 0, "num targets must be >= 1. currently: {}".format(num_targets)
        elif opt in ("-bi", "--burn-in"):
            burn_in = bool(args[i])

    N = network_size
    if N == 4:
        N_pops = 2
        pop_size = 2
    elif N == 16:
        N_pops = 4
        pop_size = 2
    elif N == 8:
        N_pops = 4
        pop_size = 2
    elif N == 2:
        N_pops = 2
        pop_size = 1
    else:
        raise NotImplementedError('N has to be in [2, 4, 16]')

    if exp_type_str in [C.ExperimentType.Synthetic.name, C.ExperimentType.SanityCheck.name]:
        for f_i in range(3, 3+num_targets):
            target_model_name = 'gif_soft_continuous_populations_model{}'.format(f_i)
            target_model = TargetModelMicroGIF.micro_gif_populations_model(random_seed=f_i, pop_size=pop_size, N_pops=N_pops)

            constants = C.Constants(learn_rate=learn_rate, train_iters=max_train_iters, N_exp=N_exp, batch_size=batch_size,
                                    tau_van_rossum=tau_van_rossum, rows_per_train_iter=rows_per_train_iter, optimiser=optimiser,
                                    initial_poisson_rate=0., loss_fn=LossFn(loss_fn).name, evaluate_step=evaluate_step,
                                    plot_flag=plot_flag, start_seed=start_seed, target_fname=target_model_name,
                                    exp_type_str=exp_type_str, silent_penalty_factor=None,
                                    norm_grad_flag=norm_grad_flag, data_path=data_path, bin_size=bin_size,
                                    burn_in=burn_in)

            gif_exp_suite.start_exp(constants=constants, model_class=microGIF, target_model=target_model)


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
