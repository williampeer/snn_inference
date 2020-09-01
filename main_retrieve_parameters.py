import sys

import Constants as C
from Models.GLIF import GLIF
from Models.LIF import LIF
from Models.LIF_ASC import LIF_ASC
from Models.LIF_R import LIF_R
from Models.LIF_R_ASC import LIF_R_ASC


def main(argv):
    print('Argument List:', str(argv))

    # Default values
    data_bin_size = 4000; target_bin_size = 1
    data_set = None
    exp_type = 'RetrieveFitted'

    learn_rate = 0.01; train_iters = 15; N_exp = 3; batch_size = 400; tau_van_rossum = 5.0
    rows_per_train_iter = 2000
    optimiser = 'Adam'
    initial_poisson_rate = 0.4
    # loss_fn = 'van_rossum_dist'  # loss_fn = 'van_rossum_dist_per_node'
    loss_fn = 'poisson_nll'  # loss_fn = 'van_rossum_dist_per_node'

    evaluate_step = 1

    fitted_model_selection = None
    path_prefix = '/Users/william/data/target_models/'
    target_models_arr = ['LIF_sleep_model.pt', 'Izhikevich_sleep_model.pt', 'GLIF_random.pt', 'Attractor_net.pt']

    fitted_model_path = '/Users/william/data/sleep_data/LIF_sleep_model/LIF_sleep_model.pt'
    # fitted_model_path = '/Users/william/data/sleep_data/Izhikevich_sleep_model/Izhikevich_sleep_model.pt'

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]

    for i, opt in enumerate(opts):
        if opt == '-h':
            print('main.py -s <script> -lr <learning-rate> -ti <training-iterations> -N <number-of-experiments> '
                  '-bs <batch-size> -tvr <van-rossum-time-constant> '
                  '-rpti <rows-per-training-iteration> -optim <optimiser> -ipr <initial-poisson-rate> '
                  '-es <evaluate-step> -fmp <fitted-model-path> -fmn <fitted-model-number>')
            sys.exit()
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
        elif opt in ("-rpti", "--rows-per-training-iteration"):
            rows_per_train_iter = int(args[i])
        elif opt in ("-optim", "--optimiser"):
            optimiser = str(args[i])
        elif opt in ("-ipr", "--initial-poisson-rate"):
            initial_poisson_rate = float(args[i])
        elif opt in ("-es", "--evaluate-step"):
            evaluate_step = int(args[i])
        elif opt in ("-fmp", "--fitted-model-path"):
            fitted_model_path = args[i]
        elif opt in ("-fms", "--fitted-model-selection"):
            fitted_model_selection = args[i]

    if fitted_model_selection is not None and fitted_model_path is None:
        sel_ind = int(fitted_model_selection)
        assert 0 < sel_ind < len(target_models_arr), \
            "index: {} for fitted models must be in the range(0, {})".format(sel_ind, len(target_models_arr))
        fitted_model_path = path_prefix + target_models_arr[sel_ind]

    constants = C.Constants(data_bin_size=data_bin_size, target_bin_size=target_bin_size, learn_rate=learn_rate,
                            train_iters=train_iters, N_exp=N_exp, batch_size=batch_size, tau_van_rossum=tau_van_rossum,
                            rows_per_train_iter=rows_per_train_iter, optimiser=optimiser,
                            initial_poisson_rate=initial_poisson_rate, loss_fn=loss_fn, data_set=data_set,
                            evaluate_step=evaluate_step, fitted_model_path=fitted_model_path)


    import retrieve_exp_suite
    # models = [LIF, LIF_R, LIF_ASC, LIF_R_ASC, GLIF]
    models = [GLIF]
    for m_class in models:
        retrieve_exp_suite.start_exp(constants=constants, model_class=m_class)


if __name__ == "__main__":
    main(sys.argv[1:])
