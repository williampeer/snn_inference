import sys
import Constants as C
from Models.Izhikevich import Izhikevich
from Models.LIF import LIF_complex, LIF
from Models.LIF_ASC import LIF_ASC
from Models.LIF_R import LIF_R
from Models.LIF_R_ASC import LIF_R_ASC
from Models.LIF_R_ASC_AT import GLIF


def main(argv):
    print('Argument List:', str(argv))

    # Default values
    data_bin_size = 4000; target_bin_size = 1
    learn_rate = 0.001; train_iters = 3; N_exp = 1; batch_size = 400; tau_van_rossum = 10.0
    input_coefficient = 1.0
    rows_per_train_iter = 800
    optimiser = 'Adam'
    # optimiser = 'SGD'
    exp_type = 'RetrieveFitted'
    # exp_type = 'DataDriven'
    # exp_type = 'Synthetic'
    # exp_type = 'SanityCheck'
    initial_poisson_rate = 0.5
    # model_type_str = IzhikevichStable.__name__
    model_type_str = GLIF.__name__
    # model_type_str = LIF.__name__
    # model_type_str = LIF_complex.__name__
    # model_type_str = BaselineSNN.__name__
    loss_fn = 'van_rossum_dist'
    # loss_fn = 'van_rossum_dist_per_node'
    data_set = None
    # data_set = 'exp147'
    evaluate_step = 1
    # fitted_model_path = None
    fitted_model_path = '/Users/william/data/sleep_data/LIF_sleep_model/LIF_sleep_model.pt'
    # fitted_model_path = '/Users/william/data/sleep_data/Izhikevich_sleep_model/Izhikevich_sleep_model.pt'

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]

    for i, opt in enumerate(opts):
        if opt == '-h':
            print('main.py -s <script> -lr <learning-rate> -ti <training-iterations> -N <number-of-experiments> '
                  '-bs <batch-size> -tvr <van-rossum-time-constant> -ic <input-coefficient> '
                  '-rpti <rows-per-training-iteration> -optim <optimiser> -ipr <initial-poisson-rate> '
                  '-mt <model-type> -lfn <loss-fn> -ds <data-set> -es <evaluate-step> -fmp <fitted-model-path>')
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
        elif opt in ("-ds", "--data-set"):
            data_set = args[i]
        elif opt in ("-es", "--evaluate-step"):
            evaluate_step = int(args[i])
        elif opt in ("-fmp", "--fitted-model-path"):
            fitted_model_path = args[i]

    constants = C.Constants(data_bin_size=data_bin_size, target_bin_size=target_bin_size, learn_rate=learn_rate,
                            train_iters=train_iters, N_exp=N_exp, batch_size=batch_size, tau_van_rossum=tau_van_rossum,
                            input_coefficient=input_coefficient, rows_per_train_iter=rows_per_train_iter,
                            optimiser=optimiser, initial_poisson_rate=initial_poisson_rate, loss_fn=loss_fn, data_set=data_set,
                            evaluate_step=evaluate_step, fitted_model_path=fitted_model_path)

    EXP_TYPE = None
    try:
        EXP_TYPE = C.ExperimentType[exp_type]
    except:
        print('Script type not supported.')

    models = [LIF, LIF_complex, Izhikevich, Izhikevich.IzhikevichStable,
              LIF_R, LIF_ASC, LIF_R_ASC, GLIF]
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
