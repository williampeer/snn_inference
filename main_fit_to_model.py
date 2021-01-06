import sys

import Constants as C
import fit_to_model_exp_suite
from Models.GLIF import GLIF
from Models.GLIF_dynamic_R_I import GLIF_dynamic_R_I
from Models.LIF import LIF
from Models.LIF_dynamic_R_I import LIF_dynamic_R_I
from TargetModels import TargetEnsembleModels


def main(argv):
    print('Argument List:', str(argv))

    # Default values
    start_seed = 0
    # exp_type_str = C.ExperimentType.SanityCheck.name
    exp_type_str = C.ExperimentType.DataDriven.name
    learn_rate = 0.02; N_exp = 5; tau_van_rossum = 100.0; plot_flag = True
    # learn_rate = 0.01; N_exp = 3; tau_van_rossum = 4.0; plot_flag = True

    max_train_iters = 30; batch_size = 400; rows_per_train_iter = 2000
    # loss_fn = 'frdvrda'
    loss_fn = 'frd'
    # loss_fn = 'frdvrd'
    # loss_fn = 'vrd'
    # loss_fn = 'vrdts'
    # loss_fn = 'vrdfrd'
    # loss_fn = 'vrdtsfrd'
    # loss_fn = 'vrdsp'

    # batch_size = 100; rows_per_train_iter = 2000; loss_fn = 'kl_div'
    # batch_size = 20; rows_per_train_iter = 4000; loss_fn = 'poisson_nll'
    # batch_size = 50; rows_per_train_iter = 4000; loss_fn = 'poisson_nll'
    # max_train_iters = 100; batch_size = 200; rows_per_train_iter = 2000; loss_fn = 'kldfrd'
    # max_train_iters = 50; batch_size = 20; rows_per_train_iter = 4000; loss_fn = 'pnllfrd'

    # max_train_iters = 40; batch_size = 400; rows_per_train_iter = 2000; loss_fn = 'free_label_vr'
    # max_train_iters = 20; batch_size = 400; rows_per_train_iter = 2000; loss_fn = 'free_label_rate_dist'
    # max_train_iters = 40; batch_size = 400; rows_per_train_iter = 2000; loss_fn = 'free_label_rate_dist_w_penalty'

    # max_train_iters = 40; batch_size = 200; rows_per_train_iter = 1600; loss_fn = 'mse'

    # optimiser = 'Adam'
    optimiser = 'SGD'
    initial_poisson_rate = 10.  # Hz

    evaluate_step = 1
    # evaluate_step = int(max(max_train_iters/10, 1))
    # data_path = None
    # prefix = '/Users/william/data/target_data/'

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
        elif opt in ("-lfn", "--loss-function"):
            loss_fn = args[i]
        elif opt in ("-sp", "--should-plot"):
            plot_flag = bool(args[i])
        elif opt in ("-trn", "--target-rate-number"):
            trn = int(args[i])
        elif opt in ("-ss", "--start-seed"):
            start_seed = int(args[i])
        elif opt in ("-et", "--experiment-type"):
            exp_type_str = args[i]

    for f_i in range(4):
        # models = [LIF, LIF_dynamic_R_I, GLIF, GLIF_dynamic_R_I]
        # models = [LIF]
        # models = [LIF_dynamic_R_I]
        # models = [GLIF]
        # models = [GLIF_dynamic_R_I]
        models = [LIF, LIF_dynamic_R_I]
        # models = [GLIF, GLIF_dynamic_R_I]
        # models = [LIF, LIF_dynamic_R_I, GLIF, GLIF_dynamic_R_I]
        # models = [LI..F, LIF_R, LIF_ASC, LIF_R_ASC, GLIF]
        for m_class in models:
            if m_class.__name__ in [LIF.__name__, LIF_dynamic_R_I.__name__]:
                target_model_name = 'lif_ensembles_model_dales_compliant_seed_{}'.format(f_i)
                target_model = TargetEnsembleModels.lif_ensembles_model_dales_compliant(random_seed=f_i)
            elif m_class.__name__ in [GLIF.__name__, GLIF_dynamic_R_I.__name__]:
                target_model_name = 'glif_ensembles_model_dales_compliant_seed_{}'.format(f_i)
                target_model = TargetEnsembleModels.glif_ensembles_model_dales_compliant(random_seed=f_i)
            else:
                raise NotImplementedError()

            constants = C.Constants(learn_rate=learn_rate, train_iters=max_train_iters, N_exp=N_exp, batch_size=batch_size,
                                    tau_van_rossum=tau_van_rossum, rows_per_train_iter=rows_per_train_iter, optimiser=optimiser,
                                    initial_poisson_rate=initial_poisson_rate, loss_fn=loss_fn, evaluate_step=evaluate_step,
                                    plot_flag=plot_flag, start_seed=start_seed, target_fname=target_model_name, exp_type_str=exp_type_str)

            fit_to_model_exp_suite.start_exp(constants=constants, model_class=m_class, target_model=target_model)

if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
