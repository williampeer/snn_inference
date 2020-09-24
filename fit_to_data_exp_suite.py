import sys

from torch import tensor as T

import Log
import data_util
from Constants import ExperimentType
from Models.GLIF import GLIF
from Models.LIF import LIF
from Models.LIF_ASC import LIF_ASC
from Models.LIF_R import LIF_R
from Models.LIF_R_ASC import LIF_R_ASC
from eval import evaluate_loss
from experiments import *
from fit import fit_mini_batches, release_computational_graph
from plot import *
from memory_profiler import profile

torch.autograd.set_detect_anomaly(True)

# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = 'cpu'
verbose = True
# ---------------------------------------


def stats_training_iterations(model_parameters, model, train_losses, test_losses, constants, logger, exp_type_str, target_parameters, exp_num, train_i):
    parameter_names = model.parameter_names
    parameter_names.append('p_\{rate\}')
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
    IO.save(model, loss={'train_losses': train_losses, 'test_losses': test_losses}, uuid=constants.UUID, fname=cur_fname)

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


# @profile
def fit_model_to_data(logger, constants, model_class, params_model, exp_num, target_parameters=False):
    node_indices, spike_times, spike_indices = data_util.load_sparse_data(constants.data_path)
    params_model['N'] = len(node_indices)

    assert constants.train_iters * constants.rows_per_train_iter <= spike_times[-1], \
        "should have enough rows. desired: {}, spikes_times[-1]: {}".format(
            constants.train_iters * constants.rows_per_train_iter, spike_times[-1])

    model = model_class(device=device, parameters=params_model)
    logger.log('initial model parameters: {}'.format(params_model), [model_class.__name__])
    poisson_input_rate = torch.tensor(constants.initial_poisson_rate, requires_grad=True)
    parameters = {}
    for p_i, key in enumerate(model.state_dict()):
        parameters[p_i] = [model.state_dict()[key].numpy()]
    parameters[p_i + 1] = [poisson_input_rate.clone().detach().numpy()]

    optim_params = list(model.parameters())
    optim_params.append(poisson_input_rate)
    optim = constants.optimiser(optim_params, lr=constants.learn_rate)

    train_losses = []; validation_losses = np.array([]); prev_spike_index = 0; train_i = 0; converged = False
    max_grads_mean = np.float(0.)
    while not converged and (train_i < constants.train_iters):
        logger.log('training iteration #{}'.format(train_i), [ExperimentType.DataDriven])
        prev_spike_index, targets = data_util.get_spike_train_matrix(index_last_step=prev_spike_index,
                                                                     advance_by_t_steps=constants.rows_per_train_iter,
                                                                     spike_times=spike_times, spike_indices=spike_indices,
                                                                     node_numbers=node_indices)

        avg_train_loss, abs_grads_mean = fit_mini_batches(model, gen_inputs=None, target_spiketrain=targets,
                                                          poisson_input_rate=poisson_input_rate, optimiser=optim,
                                                          constants=constants, train_i=train_i, logger=logger)
        logger.log(parameters=[avg_train_loss, abs_grads_mean])
        train_losses.append(avg_train_loss)

        cur_params = model.state_dict()
        logger.log('current parameters {}'.format(cur_params))
        for p_i, key in enumerate(cur_params):
            parameters[p_i].append(cur_params[key].clone().detach().numpy())
        parameters[p_i + 1].append(poisson_input_rate.clone().detach().numpy())

        max_grads_mean = np.max((max_grads_mean, abs_grads_mean))
        converged = abs(abs_grads_mean) <= 0.2 * abs(max_grads_mean)

        # gradients = []
        # for param_i, param in enumerate(list(model.parameters())):
        #     logger.log('parameter #{} gradient: {}'.format(param_i, param.grad))
        #     gradients.append(param.grad)
        # gradients.append(current_rate.grad)

        # if train_i % constants.evaluate_step == 0 or (converged or (train_i+1 >= constants.train_iters)):
        prev_spike_index, targets = data_util.get_spike_train_matrix(index_last_step=prev_spike_index,
                                                                     advance_by_t_steps=constants.rows_per_train_iter,
                                                                     spike_times=spike_times,
                                                                     spike_indices=spike_indices,
                                                                     node_numbers=node_indices)
        validation_inputs = poisson_input(rate=poisson_input_rate, t=constants.rows_per_train_iter, N=model.N)
        validation_loss = evaluate_loss(model, inputs=validation_inputs, target_spiketrain=targets, uuid=constants.UUID,
                                        tau_van_rossum=constants.tau_van_rossum, label='train i: {}'.format(train_i),
                                        exp_type=ExperimentType.DataDriven, train_i=train_i, exp_num=exp_num, constants=constants)
        logger.log(parameters=['validation loss', validation_loss])
        validation_losses = np.concatenate((validation_losses, np.asarray([validation_loss])))

        release_computational_graph(model, poisson_input_rate, validation_inputs)
        targets = None; validation_inputs = None; validation_loss = None
        train_i += 1

    # model.load_state_dict(prev_state_dict)
    stats_training_iterations(parameters, model, train_losses, validation_losses, constants, logger, ExperimentType.DataDriven.name,
                              target_parameters=target_parameters, exp_num=exp_num, train_i=train_i)
    model = None
    return parameters, train_losses, validation_losses, train_i


# @profile
def run_exp_loop(logger, constants, model_class, free_model_parameters, target_parameters=False):
    all_recovered_params = {}
    for exp_i in range(constants.N_exp):
        torch.manual_seed(exp_i)
        np.random.seed(exp_i)

        # TODO: uniform for intervals across trials? for comparison w gradient-free nevergrad approaches.
        params_model = randomise_parameters(free_model_parameters, coeff=torch.tensor(0.1))

        recovered_parameters, train_losses, test_losses, train_i = \
            fit_model_to_data(logger, constants, model_class, params_model, exp_num=exp_i, target_parameters=target_parameters)

        if train_i >= constants.train_iters:
            print('DID NOT CONVERGE FOR SEED, CONTINUING ON TO NEXT SEED. exp_i: {}, train_i: {}, train_losses: {}, test_losses: {}'
                  .format(exp_i, train_i, train_losses, test_losses))

        for p_i, key in enumerate(recovered_parameters):
            if exp_i == 0:
                all_recovered_params[key] = [recovered_parameters[key]]
            else:
                all_recovered_params[key].append(recovered_parameters[key])

    parameter_names = model_class.parameter_names
    parameter_names.append('p_\{rate\}')
    plot_all_param_pairs_with_variance(all_recovered_params,
                                       uuid=constants.UUID,
                                       exp_type=ExperimentType.RetrieveFitted.name,
                                       target_params=target_parameters,
                                       param_names=parameter_names,
                                       custom_title="Average inferred parameters across experiments [{}, {}]".format(
                                           model_class.__name__, constants.optimiser),
                                       logger=logger, fname='all_inferred_params_{}'.format(model_class.__name__))


def start_exp(constants, model_class, target_parameters=False):
    log_fname = model_class.__name__ + '{}_lr_{}_batchsize_{}_trainiters_{}_rowspertrainiter_{}_uuid_{}'.\
        format(ExperimentType.DataDriven.name, '{:1.3f}'.format(constants.learn_rate).replace('.', '_'), constants.batch_size,
               constants.train_iters, constants.rows_per_train_iter, constants.UUID)
    logger = Log.Logger(log_fname)
    logger.log('Starting exp. with listed hyperparameters.', [constants.__str__()])

    if model_class in [LIF, LIF_R, LIF_ASC, LIF_R_ASC, GLIF]:
        free_parameters = {'w_mean': 0.3, 'w_var': 0.5, 'C_m': 1.5, 'G': 0.8, 'R_I': 18., 'E_L': -60.,
                           'delta_theta_s': 25., 'b_s': 0.4, 'f_v': 0.14, 'delta_V': 12., 'f_I': 0.4, 'I_A': 1.,
                           'b_v': 0.5, 'a_v': 0.5, 'theta_inf': -25.}
    else:
        logger.log('Model class not supported.')
        sys.exit(1)

    run_exp_loop(logger, constants, model_class, free_parameters, target_parameters)
