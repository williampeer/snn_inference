import sys

import Constants as C
import data_util
import exp_suite
from Models.GLIF import GLIF
from Models.LIF import LIF
from Models.LIF_R import LIF_R
from Models.LIF_R_ASC import LIF_R_ASC
from Models.LIF_R_weights_only import LIF_R_weights_only
from Models.LowerDim.GLIF_lower_dim import GLIF_lower_dim
from Models.LowerDim.GLIF_soft_lower_dim import GLIF_soft_lower_dim
from Models.LowerDim.LIF_R_ASC_lower_dim import LIF_R_ASC_lower_dim
from Models.LowerDim.LIF_R_ASC_soft_lower_dim import LIF_R_ASC_soft_lower_dim
from Models.LowerDim.LIF_R_lower_dim import LIF_R_lower_dim
from Models.LowerDim.LIF_R_soft_lower_dim import LIF_R_soft_lower_dim
from Models.Sigmoidal.GLIF_soft import GLIF_soft
from Models.Sigmoidal.GLIF_soft_positive_weights import GLIF_soft_positive_weights
from Models.Sigmoidal.LIF_R_ASC_soft import LIF_R_ASC_soft
from Models.Sigmoidal.LIF_R_soft import LIF_R_soft
from Models.Sigmoidal.LIF_R_soft_weights_only import LIF_R_soft_weights_only
from Models.microGIF import microGIF
from TargetModels import TargetModels, TargetModelsSoft, TargetModelMicroGIF
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
    # learn_rate = 0.05; N_exp = 5; tau_van_rossum = 4.0; plot_flag = True
    # max_train_iters = 10; batch_size = 1000; rows_per_train_iter = 2000
    learn_rate = 0.02; N_exp = 1; tau_van_rossum = 20.0; plot_flag = True
    # Run 100 with lr 0.01 and 0.02
    max_train_iters = 40
    num_targets = 1
    # Q: Interval size effect on loss curve and param retrieval for both lfns
    interval_size = 4800
    batch_size = interval_size; rows_per_train_iter = interval_size
    # bin_size = int(interval_size/10)  # for RPH
    bin_size = 100  # ms
    burn_in = False
    # burn_in = True
    # batch_size = 2000; rows_per_train_iter = 8000
    # learn_rate = 0.01; N_exp = 3; tau_van_rossum = 4.0; plot_flag = True
    # loss_fn = 'frd'
    loss_fn = 'vrd'
    # loss_fn = 'FF'
    # loss_fn = 'CV'
    # loss_fn = 'PCC'
    # loss_fn = 'rfh'
    # loss_fn = 'rph'
    # loss_fn = 'kl_div'
    # loss_fn = None
    # silent_penalty_factor = 10.0
    silent_penalty_factor = None

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
    # optimiser = 'RMSprop'
    initial_poisson_rate = 10.  # Hz
    # network_size = 2
    # network_size = 4
    network_size = 8
    # network_size = 16

    evaluate_step = 10
    # evaluate_step = int(max(max_train_iters/10, 1))
    data_path = None
    # data_path = data_util.prefix + data_util.path + 'target_model_spikes_GLIF_seed_4.mat'
    # data_path = data_util.prefix + data_util.path + 'target_model_spikes_GLIF_seed_4_duration_300000.mat'
    # data_path = data_util.prefix + data_util.path + 'target_model_spikes_GLIF_seed_4_N_3_duration_300000.mat'
    # data_path = data_util.prefix + data_util.path + 'target_model_spikes_GLIF_N_3_seed_4_duration_1800000.mat'
    # data_path = data_util.prefix + data_util.path + 'target_model_spikes_GLIF_N_12_seed_4_duration_900000.mat'
    # data_path = data_util.prefix + data_util.path + 'target_model_spikes_LIF_R_N_3_seed_4_duration_900000.mat'  # !!!!!

    # model_type = None
    # model_type = 'GLIF'
    model_type = 'LIF'
    # model_type = 'microGIF'
    # model_type = 'LIF_weights_only'
    # model_type = 'LIF_fixed_weights'
    # model_type = 'LIF_soft'
    # model_type = 'LIF_soft_weights_only'
    # model_type = 'LIF_ASC'
    # model_type = 'LIF_weights_only'
    # model_type = 'LIF_R_soft'
    # model_type = 'LIF_R_soft_weights_only'
    # model_type = 'GLIF_soft_positive_weights'
    # model_type = 'LIF_R_ASC'
    # model_type = 'LIF_R_ASC_soft'
    # model_type = 'LIF_R_ASC_soft_ReLu'
    # model_type = 'GLIF_soft'
    # model_type = 'GLIF_soft_lower_dim'
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
        elif opt in ("-ipr", "--initial-poisson-rate"):
            initial_poisson_rate = float(args[i])
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
        elif opt in ("-spf", "--silent-penalty-factor"):
            silent_penalty_factor = float(args[i])
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

    all_models = [LIF, LIF_R, LIF_R_ASC, GLIF,
                  LIF_R_soft, LIF_R_ASC_soft, GLIF_soft,
                  LIF_R_weights_only, LIF_R_soft_weights_only,
                  GLIF_soft_positive_weights, GLIF_soft_lower_dim,
                  LIF_R_ASC_soft_lower_dim, LIF_R_soft_lower_dim,
                  GLIF_lower_dim, LIF_R_ASC_lower_dim, LIF_R_lower_dim,
                  microGIF]
    # models = [LIF_HS_17]
    # models = [LIF, LIF_R, LIF_ASC, LIF_R_ASC, GLIF]
    # models = [LIF_soft_weights_only, LIF_R_soft, LIF_ASC_soft, LIF_R_ASC_soft, GLIF_soft]
    # models = [LIF, LIF_weights_only, LIF_soft, LIF_soft_weights_only]
    # models = [LIF, LIF_soft, LIF_weights_only, LIF_soft_weights_only]
    # models = [LIF, LIF_R, LIF_ASC, LIF_R_ASC, GLIF]
    # models = [LIF, LIF_fixed_weights, LIF_weights_only]
    # models = [LIF_R_soft_weights_only, LIF_R_soft, LIF_R_ASC_soft, GLIF_soft]
    # models = [LIF_R_soft_weights_only, LIF_R_soft, LIF_R_ASC_soft, GLIF_soft, LIF_R_weights_only, LIF_R, LIF_R_ASC, GLIF, GLIF_soft_positive_weights]
    # models = [LIF_R, LIF_R_ASC, GLIF]
    # models = [GLIF_soft_lower_dim, LIF_R_ASC_soft_lower_dim, LIF_R_soft_lower_dim, GLIF_lower_dim, LIF_R_ASC_lower_dim, LIF_R_lower_dim]
    models = [GLIF_soft_lower_dim, LIF_R_soft_lower_dim, GLIF_lower_dim, LIF_R_ASC_lower_dim, LIF_R_lower_dim]
    # models = [LIF_R_ReLu_weights_only, LIF_R_ReLu, LIF_R_ASC_ReLu, GLIF_ReLu]
    # models = [LIF_R_weights_only, LIF_R, LIF_R_ASC, GLIF]
    # models = [LIF_R_soft_weights_only, LIF_R_soft, LIF_R_ASC_soft, GLIF_soft,
    #           LIF_R_soft_ReLu_weights_only, LIF_R_soft_ReLu, LIF_R_ASC_soft_ReLu, GLIF_soft_ReLu]
    # models = [LIF_weights_only, LIF_fixed_weights, LIF_R]

    if loss_fn is None:
        loss_functions = [LossFn.FIRING_RATE_DIST.name, LossFn.VAN_ROSSUM_DIST.name, LossFn.RATE_PCC_HYBRID.name]
                          # LossFn.PEARSON_CORRELATION_COEFFICIENT.name,
                          # LossFn.FANO_FACTOR_DIST.name,
                          # LossFn.RATE_FANO_HYBRID.name,
                          # LossFn.CV_DIST.name]
                          # LossFn.KL_DIV.name]
                          # LossFn.MSE.name]
    else:
        loss_functions = [LossFn(loss_fn).name]
    if model_type is not None and model_type in str(all_models):
        for m in all_models:
            if m.__name__ == model_type:
                models = [m]
        if len(models) > 1:
            print('Did not find supplied model type. Iterating over all implemented models..')

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

    for m_class in models:
        for loss_fn in loss_functions:
            if exp_type_str in [C.ExperimentType.Synthetic.name, C.ExperimentType.SanityCheck.name]:
                for f_i in range(3, 3+num_targets):
                    # if m_class.__name__ in [LIF_R.__name__, LIF_R_weights_only.__name__, LIF_R_lower_dim.__name__]:
                    #     target_model_name = 'lif_r_ensembles_model_dales_compliant_seed_{}'.format(f_i)
                    #     target_model = TargetModels.lif_r_continuous_ensembles_model_dales_compliant(random_seed=f_i, pop_size=pop_size, N_pops=N_pops)
                    # elif m_class.__name__ in [LIF_R_soft.__name__, LIF_R_soft_weights_only.__name__, LIF_R_soft_lower_dim.__name__]:
                    #     target_model_name = 'lif_r_soft_ensembles_model_dales_compliant_seed_{}'.format(f_i)
                    #     target_model = TargetModelsSoft.lif_r_soft_continuous_ensembles_model_dales_compliant(random_seed=f_i, pop_size=pop_size, N_pops=N_pops)
                    #
                    # elif m_class.__name__ in [LIF_R_ASC.__name__, LIF_R_ASC_soft.__name__, LIF_R_ASC_soft_lower_dim.__name__, LIF_R_ASC_lower_dim.__name__]:
                    #     target_model_name = 'lif_r_asc_ensembles_model_dales_compliant_seed_{}'.format(f_i)
                    #     target_model = TargetModels.lif_r_asc_continuous_ensembles_model_dales_compliant(random_seed=f_i, pop_size=pop_size, N_pops=N_pops)
                    # elif m_class.__name__ in [GLIF.__name__, GLIF_lower_dim.__name__]:
                    #     target_model_name = 'glif_ensembles_model_dales_compliant_seed_{}'.format(f_i)
                    #     target_model = TargetModels.glif_continuous_ensembles_model_dales_compliant(random_seed=f_i, pop_size=pop_size, N_pops=N_pops)

                    if m_class.__name__ in [GLIF_soft.__name__, GLIF_soft_lower_dim.__name__]:
                        target_model_name = 'glif_soft_ensembles_model_dales_compliant_seed_{}'.format(f_i)
                        target_model = TargetModelsSoft.glif_soft_continuous_ensembles_model_dales_compliant(random_seed=f_i, pop_size=pop_size, N_pops=N_pops)
                    elif m_class.__name__ in [LIF.__name__]:
                        target_model_name = 'lif_pop_model_{}'.format(f_i)
                        target_model = TargetModelsSoft.lif_pop_model(random_seed=f_i, pop_size=pop_size, N_pops=N_pops)
                    elif m_class.__name__ in [microGIF.__name__]:
                        target_model_name = 'micro_gif_populations_model_{}'.format(f_i)
                        target_model = TargetModelMicroGIF.micro_gif_populations_model(random_seed=f_i, pop_size=pop_size, N_pops=N_pops)

                    else:
                        raise NotImplementedError()

                    constants = C.Constants(learn_rate=learn_rate, train_iters=max_train_iters, N_exp=N_exp, batch_size=batch_size,
                                            tau_van_rossum=tau_van_rossum, rows_per_train_iter=rows_per_train_iter, optimiser=optimiser,
                                            initial_poisson_rate=initial_poisson_rate, loss_fn=loss_fn, evaluate_step=evaluate_step,
                                            plot_flag=plot_flag, start_seed=start_seed, target_fname=target_model_name,
                                            exp_type_str=exp_type_str, silent_penalty_factor=silent_penalty_factor,
                                            norm_grad_flag=norm_grad_flag, data_path=data_path, bin_size=bin_size,
                                            burn_in=burn_in)

                    exp_suite.start_exp(constants=constants, model_class=m_class, target_model=target_model)

            # elif exp_type_str == C.ExperimentType.DataDriven.name:
            #     for f_i in range(3, 3+num_targets):
            #         # only for target_parameters
            #         if m_class.__name__ in [LIF_R.__name__, LIF_R_soft.__name__, LIF_R_weights_only.__name__,
            #                                 LIF_R_soft_weights_only.__name__]:
            #             target_model = TargetModels.lif_r_continuous_ensembles_model_dales_compliant(random_seed=f_i, pop_size=pop_size, N_pops=N_pops)
            #         elif m_class.__name__ in [LIF_R_ASC.__name__, LIF_R_ASC_soft.__name__]:
            #             target_model = TargetModels.lif_r_asc_continuous_ensembles_model_dales_compliant(random_seed=f_i, pop_size=pop_size, N_pops=N_pops)
            #         elif m_class.__name__ in [GLIF.__name__, GLIF_soft.__name__]:
            #             target_model = TargetModels.glif_continuous_ensembles_model_dales_compliant(random_seed=f_i, pop_size=pop_size, N_pops=N_pops)
            #         else:
            #             raise NotImplementedError()
            #
            #         data_path = data_util.prefix + data_util.path + 'target_model_spikes_{}_N_{}_seed_{}_duration_{}' \
            #             .format(target_model.name(), network_size, f_i, 15 * 60 * 1000)
            #
            #         constants = C.Constants(learn_rate=learn_rate, train_iters=max_train_iters, N_exp=N_exp, batch_size=batch_size,
            #                                 tau_van_rossum=tau_van_rossum, rows_per_train_iter=rows_per_train_iter, optimiser=optimiser,
            #                                 initial_poisson_rate=initial_poisson_rate, loss_fn=loss_fn, evaluate_step=evaluate_step,
            #                                 plot_flag=plot_flag, start_seed=start_seed, exp_type_str=exp_type_str,
            #                                 silent_penalty_factor=silent_penalty_factor, norm_grad_flag=norm_grad_flag,
            #                                 data_path=data_path)
            #
            #         exp_suite.start_exp(constants=constants, model_class=m_class, target_model=target_model)


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
