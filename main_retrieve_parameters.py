import sys

import Constants as C
from Models.GLIF import GLIF
from TargetModels import TargetModels


def main(argv):
    print('Argument List:', str(argv))

    # Default values
    learn_rate = 0.01; N_exp = 3; tau_van_rossum = 4.0

    train_iters = 40; batch_size = 200; rows_per_train_iter = 400; loss_fn = 'poisson_nll'
    # train_iters = 40; batch_size = 100; rows_per_train_iter = 400; loss_fn = 'van_rossum_dist'
    # train_iters = 20; batch_size = 200; rows_per_train_iter = 800; loss_fn = 'van_rossum_dist'

    optimiser = 'Adam'
    initial_poisson_rate = 0.6

    evaluate_step = 1

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

    constants = C.Constants(learn_rate=learn_rate, train_iters=train_iters, N_exp=N_exp, batch_size=batch_size,
                            tau_van_rossum=tau_van_rossum, rows_per_train_iter=rows_per_train_iter, optimiser=optimiser,
                            initial_poisson_rate=initial_poisson_rate, loss_fn=loss_fn, evaluate_step=evaluate_step)

    # free_parameters = {'w_mean': 0.3, 'w_var': 0.5, 'C_m': 1.5, 'G': 0.8, 'R_I': 20., 'E_L': -60., 'delta_theta_s': 25.,
    #                    'b_s': 0.4, 'f_v': 0.14, 'delta_V': 12., 'f_I': 0.4, 'I_A': 1., 'b_v': 0.5, 'a_v': 0.5, 'theta_inf': -25.}
    # gen_model = GLIF(device='cpu', parameters=free_parameters, N=12)
    # gen_model = torch.load(constants.fitted_model_path)['model']
    # gen_model = SleepModelWrappers.glif_sleep_model()
    # gen_model = TargetModels.glif_recurrent_net()
    gen_model = TargetModels.random_glif_model()

    import retrieve_exp_suite
    # models = [LIF, LIF_R, LIF_ASC, LIF_R_ASC, GLIF]
    models = [GLIF]
    for m_class in models:
        retrieve_exp_suite.start_exp(constants=constants, model_class=m_class, gen_model=gen_model)


if __name__ == "__main__":
    main(sys.argv[1:])
