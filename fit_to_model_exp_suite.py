import sys

import Log
from Constants import ExperimentType
from IO import save_poisson_rates
from Models.GLIF import GLIF
from Models.LIF import LIF
from Models.LIF_ASC import LIF_ASC
from Models.LIF_R import LIF_R
from Models.LIF_R_ASC import LIF_R_ASC
from experiments import generate_synthetic_data, draw_from_uniform, zip_dicts, release_computational_graph
from fit import fit_mini_batches
from plot import *

torch.autograd.set_detect_anomaly(True)

# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = 'cpu'
verbose = True
# ---------------------------------------


def stats_training_iterations(model_parameters, model, poisson_rate, train_losses, test_losses, constants, logger, exp_type_str, target_parameters, exp_num, train_i):
    if constants.plot_flag:
        parameter_names = model.parameter_names
        parameter_names.append('p_rate')
        plot_parameter_inference_trajectories_2d(model_parameters,
                                                 uuid=constants.UUID,
                                                 exp_type=exp_type_str,
                                                 target_params=target_parameters,
                                                 param_names=parameter_names,
                                                 custom_title='Inferred parameters across training iterations',
                                                 fname='inferred_param_trajectories_{}_exp_num_{}_train_iters_{}'
                                                 .format(model.__class__.__name__, exp_num, train_i),
                                                 logger=logger)
        plot_losses(training_loss=train_losses, test_loss=test_losses, test_loss_step=constants.evaluate_step, uuid=constants.UUID, exp_type=exp_type_str,
                    custom_title='Loss ({}, {}, lr={})'.format(model.__class__.__name__, constants.optimiser.__name__, constants.learn_rate),
                    fname='training_and_test_loss_exp_{}_loss_fn_{}'.format(exp_num, constants.loss_fn))

    logger.log('train_losses: #{}'.format(train_losses))
    mean_test_loss = torch.mean(torch.tensor(test_losses)).clone().detach().numpy()
    logger.log('test_losses: #{}'.format(test_losses), ['mean test loss: {}'.format(mean_test_loss)])

    cur_fname = '{}_exp_num_{}_data_set_{}_mean_loss_{:.3f}_uuid_{}'.format(model.__class__.__name__, exp_num, constants.data_set, mean_test_loss, constants.UUID)
    IO.save(model, rate=poisson_rate, loss={'train_losses': train_losses, 'test_losses': test_losses}, uuid=constants.UUID, fname=cur_fname)

    del model, mean_test_loss


def convergence_check(validation_losses):
    if len(validation_losses) <= 1:
        return False

    val_diff = validation_losses[-1] - validation_losses[-2]
    return val_diff >= 0.


def overall_gradients_mean(gradients, train_i, loss_fn):
    mean_logger = Log.Logger('gradients_mean_log')
    full_logger = Log.Logger('gradients_full_log')

    avg_grads = []
    for i, grads in enumerate(gradients):
        avg_grads.append(torch.mean(grads))
    overall_mean = torch.mean(torch.tensor(avg_grads))
    mean_logger.log('avg_grads: {}, train_i: {}, loss_fn: {}'.format(avg_grads, train_i, loss_fn))
    mean_logger.log('overall_mean: {}'.format(overall_mean))

    full_logger.log('train_i: {}, loss_fn: {}, gradients'.format(train_i, loss_fn), gradients)
    return float(overall_mean.clone().detach())


def fit_model_to_target_model(logger, constants, model_class, params_model, exp_num, target_model, target_parameters):
    params_model['N'] = target_model.N
    model = model_class(device=device, N=target_model.N, parameters=params_model)
    logger.log('initial model parameters: {}'.format(params_model), [model_class.__name__])
    poisson_input_rate = torch.tensor(constants.initial_poisson_rate, requires_grad=True)
    poisson_input_rate.clamp(1., 40.)
    parameters = {}
    for p_i, key in enumerate(model.state_dict()):
        parameters[p_i] = [model.state_dict()[key].numpy()]
    # parameters[p_i + 1] = [poisson_input_rate.clone().detach().numpy()]
    poisson_rates = []
    poisson_rates.append(poisson_input_rate.clone().detach().numpy())

    optim_params = list(model.parameters())
    optim_params.append(poisson_input_rate)
    optim = constants.optimiser(optim_params, lr=constants.learn_rate)

    train_losses = []; validation_losses = np.array([]); prev_spike_index = 0; train_i = 0; converged = False
    max_grads_mean = np.float(0.)
    while not converged and (train_i < constants.train_iters):
        logger.log('training iteration #{}'.format(train_i), [ExperimentType.DataDriven])

        targets = generate_synthetic_data(target_model, poisson_rate=constants.initial_poisson_rate, t=constants.rows_per_train_iter)

        avg_train_loss, abs_grads_mean, last_loss = fit_mini_batches(model, gen_inputs=None, target_spiketrain=targets,
                                                                     poisson_input_rate=poisson_input_rate, optimiser=optim,
                                                                     constants=constants, train_i=train_i, logger=logger)
        logger.log(parameters=[avg_train_loss, abs_grads_mean])
        train_losses.append(avg_train_loss)

        cur_params = model.state_dict()
        logger.log('current parameters {}'.format(cur_params))
        for p_i, key in enumerate(cur_params):
            parameters[p_i].append(cur_params[key].clone().detach().numpy())
        # parameters[p_i + 1].append(poisson_input_rate.clone().detach().numpy())
        poisson_rates.append(poisson_input_rate.clone().detach().numpy())

        # if train_i % constants.evaluate_step == 0 or (converged or (train_i+1 >= constants.train_iters)):
        # targets = generate_synthetic_data(target_model, poisson_rate=constants.initial_poisson_rate, t=constants.rows_per_train_iter)

        # validation_loss = evaluate_loss(model, inputs=None, target_spiketrain=targets, uuid=constants.UUID,
        #                                 tau_van_rossum=constants.tau_van_rossum, label='train i: {}'.format(train_i),
        #                                 exp_type=ExperimentType.DataDriven, train_i=train_i, exp_num=exp_num, constants=constants)
        validation_loss = last_loss
        logger.log(parameters=['validation loss', validation_loss])
        validation_losses = np.concatenate((validation_losses, np.asarray([validation_loss])))

        max_grads_mean = np.max((max_grads_mean, abs_grads_mean))
        # converged = abs(abs_grads_mean) <= 0.2 * abs(max_grads_mean) and validation_loss <= 0.8 * np.max(validation_losses)
        converged = abs(abs_grads_mean) <= 0.1 * abs(max_grads_mean)  # and validation_loss <= 0.8 * np.max(validation_losses)

        release_computational_graph(model, poisson_input_rate)
        targets = None; validation_loss = None
        train_i += 1

    stats_training_iterations(parameters, model, poisson_input_rate, train_losses, validation_losses, constants, logger,
                              ExperimentType.DataDriven.name, target_parameters=target_parameters, exp_num=exp_num, train_i=train_i)
    final_model_parameters = {}
    for p_i, key in enumerate(model.state_dict()):
        final_model_parameters[p_i] = [model.state_dict()[key].numpy()]
    model = None
    return final_model_parameters, train_losses, validation_losses, train_i, poisson_rates


def run_exp_loop(logger, constants, model_class, free_model_parameters, static_parameters, target_model):
    target_parameters = {}
    for param_i, key in enumerate(target_model.state_dict()):
        target_parameters[param_i - 1] = target_model.state_dict()[key].clone().detach().numpy()

    recovered_param_per_exp = {}; poisson_rate_per_exp = []
    for exp_i in range(constants.start_seed, constants.start_seed+constants.N_exp):
        torch.manual_seed(exp_i)
        np.random.seed(exp_i)
        target_model.load_state_dict(target_model.state_dict())

        num_neurons = int(target_model.v.shape[0])
        params_model = draw_from_uniform(free_model_parameters, model_class.parameter_init_intervals, num_neurons)
        params_model = zip_dicts(params_model, static_parameters)

        recovered_parameters, train_losses, test_losses, train_i, poisson_rates = \
            fit_model_to_target_model(logger, constants, model_class, params_model, exp_num=exp_i, target_model=target_model, target_parameters=target_parameters)
        logger.log('poisson rates for exp {}'.format(exp_i), poisson_rates)

        if train_i >= constants.train_iters:
            print('DID NOT CONVERGE FOR SEED, CONTINUING ON TO NEXT SEED. exp_i: {}, train_i: {}, train_losses: {}, test_losses: {}'
                  .format(exp_i, train_i, train_losses, test_losses))

        for p_i, key in enumerate(recovered_parameters):
            if exp_i == constants.start_seed:
                recovered_param_per_exp[key] = [recovered_parameters[key]]
            else:
                recovered_param_per_exp[key].append(recovered_parameters[key])
        poisson_rate_per_exp.append(poisson_rates[-1])

    logger.log('poisson_rate_per_exp', poisson_rate_per_exp)
    save_poisson_rates(poisson_rate_per_exp, uuid=constants.UUID, fname='poisson_rates_per_exp.pt')
    parameter_names = model_class.parameter_names
    parameter_names.append('p_rate')
    if constants.plot_flag:
        plot_all_param_pairs_with_variance(recovered_param_per_exp,
                                       uuid=constants.UUID,
                                       exp_type=ExperimentType.DataDriven.name,
                                       target_params=target_parameters,
                                       param_names=parameter_names,
                                       custom_title="Average inferred parameters across experiments [{}, {}]".format(
                                           model_class.__name__, constants.optimiser),
                                       logger=logger, fname='all_inferred_params_{}'.format(model_class.__name__))


def start_exp(constants, model_class, target_model):
    log_fname = model_class.__name__ + '{}_lr_{}_batchsize_{}_trainiters_{}_rowspertrainiter_{}_uuid_{}'.\
        format(ExperimentType.DataDriven.name, '{:1.3f}'.format(constants.learn_rate).replace('.', '_'), constants.batch_size,
               constants.train_iters, constants.rows_per_train_iter, constants.UUID)
    logger = Log.Logger(log_fname)
    logger.log('Starting exp. with listed hyperparameters.', [constants.__str__()])

    if model_class in [LIF, LIF_R, LIF_ASC, LIF_R_ASC, GLIF]:
        # free_parameters = {'C_m': 1.5, 'G': 0.8, 'E_L': -60., 'delta_theta_s': 25., 'b_s': 0.4, 'f_v': 0.14,
        #                    'delta_V': 12., 'f_I': 0.4, 'I_A': 1., 'b_v': 0.5, 'a_v': 0.5, 'theta_inf': -25.}
        free_parameters = {'C_m', 'G', 'E_L', 'delta_theta_s', 'b_s', 'f_v', 'delta_V', 'f_I', 'I_A', 'b_v', 'a_v', 'theta_inf'}
        static_parameters = {'R_I': 18.}
    else:
        logger.log('Model class not supported.')
        sys.exit(1)

    run_exp_loop(logger, constants, model_class, free_parameters, static_parameters, target_model)
