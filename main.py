import sys
import Constants as C
from Models import LIF, Izhikevich, BaselineSNN


def main(argv):
    # print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(argv))

    # Default values
    data_bin_size = 4000; target_bin_size = 1
    learn_rate = 0.05; train_iters = 12; N_exp = 1; batch_size = 200; tau_van_rossum = 3.0
    input_coefficient = 1.0
    # rows_per_train_iter = data_bin_size
    rows_per_train_iter = 1600
    optimiser = 'Adam'
    # optimiser = 'SGD'
    exp_type = 'Synthetic'
    initial_poisson_rate = 0.5
    model_type_str = Izhikevich.IzhikevichStable.__name__
    # model_type_str = Izhikevich.Izhikevich.__name__
    # model_type_str = Izhikevich.IzhikevichWeightsOnly.__name__
    # model_type_str = LIF.LIF.__name__
    # model_type_str = BaselineSNN.BaselineSNN.__name__
    # loss_fn = 'van_rossum_dist'
    loss_fn = 'van_rossum_dist_per_node'
    # loss_fn = 'van_rossum_squared_per_node'
    # loss_fn = 'mse_per_node'

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]

    for i, opt in enumerate(opts):
        if opt == '-h':
            print('main.py -s <script> -lr <learning-rate> -ti <training-iterations> -N <number-of-experiments> '
                  '-bs <batch-size> -tvr <van-rossum-time-constant> -ic <input-coefficient> '
                  '-rpti <rows-per-training-iteration> -optim <optimiser> -ipr <initial-poisson-rate> '
                  '-mt <model-type> -lfn <loss-fn>')
            sys.exit()
        elif opt in ("-s", "--script"):
            exp_type = args[i]
        elif opt in ("-lr", "--learning-rate"):
            learn_rate = float(args[i])
        elif opt in ("-ti", "--training-iterations"):
            train_iters = int(args[i])
        elif opt in ("-N", "--numbers-of-experiments"):
            N_exp = int(args[i])
        elif opt in ("-bs", "--batch-size"):
            batch_size = int(args[i])
        elif opt in ("-tvr", "--van-rossum-time-constant"):
            tau_van_rossum = float(args[i])
        elif opt in ("-ic", "--input-coefficient"):
            input_coefficient = float(args[i])
        elif opt in ("-rpti", "--rows-per-training-iteration"):
            rows_per_train_iter = int(args[i])
        elif opt in ("-optim", "--optimiser"):
            optimiser = str(args[i])
        elif opt in ("-ipr", "--initial-poisson-rate"):
            initial_poisson_rate = float(args[i])
        elif opt in ("-mt", "--model-type"):
            model_type_str = args[i]
        elif opt in ("-lfn", "--loss-fn"):
            loss_fn = args[i]

    constants = C.Constants(data_bin_size=data_bin_size, target_bin_size=target_bin_size, learn_rate=learn_rate,
                            train_iters=train_iters, N_exp=N_exp, batch_size=batch_size, tau_van_rossum=tau_van_rossum,
                            input_coefficient=input_coefficient, rows_per_train_iter=rows_per_train_iter,
                            optimiser=optimiser, initial_poisson_rate=initial_poisson_rate, loss_fn=loss_fn)

    EXP_TYPE = None
    try:
        EXP_TYPE = C.ExperimentType[exp_type]
    except:
        print('Script type not supported.')

    models = [BaselineSNN.BaselineSNN, LIF.LIF, Izhikevich.Izhikevich, Izhikevich.IzhikevichStable, Izhikevich.IzhikevichWeightsOnly]
    model_class = None
    for _, c in enumerate(models):
        if model_type_str == c.__name__:
            model_class = c
            break
    if model_class is None:
        print('Model type not supported.')
        sys.exit(1)


    import exp_suite
    exp_suite.start_exp(constants=constants, model_class=model_class, experiment_type=EXP_TYPE)


if __name__ == "__main__":
    main(sys.argv[1:])
