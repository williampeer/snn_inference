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
from fit import *
from plot import *

torch.autograd.set_detect_anomaly(True)

# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = 'cpu'
verbose = True
# ---------------------------------------


def stats_training_iterations(model_parameters, model, train_losses, test_losses, constants, logger, exp_type_str, target_parameters, exp_num, train_i):
    plot_parameter_inference_trajectories_2d(model_parameters,
                                             uuid=constants.UUID,
                                             exp_type=exp_type_str,
                                             target_params=target_parameters,
                                             param_names=model.parameter_names,
                                             custom_title='Inferred parameters across training iterations',
                                             fname='inferred_param_trajectories_{}_exp_num_{}_train_iters_{}'
                                             .format(model.__class__.__name__, exp_num, train_i),
                                             logger=logger)

    plot_losses(training_loss=train_losses, test_loss=test_losses, test_loss_step=constants.evaluate_step, uuid=constants.UUID, exp_type=exp_type_str,
                custom_title='Loss ({}, {}, lr={})'.format(model.__class__.__name__, constants.optimiser.__name__, constants.learn_rate),
                fname='training_and_test_loss_exp_{}_loss_fn_{}'.format(exp_num, constants.loss_fn))

    logger.log('', 'train_losses: #{}'.format(train_losses))
    mean_test_loss = torch.mean(torch.tensor(test_losses)).data
    logger.log(['mean test loss: {}'.format(mean_test_loss)], 'test_losses: #{}'.format(test_losses))

    cur_fname = '{}_exp_num_{}_data_set_{}_mean_loss_{:.3f}_uuid_{}'.format(model.__class__.__name__, exp_num, constants.data_set, mean_test_loss, constants.UUID)
    IO.save(model, loss={'train_losses': train_losses, 'test_losses': test_losses}, uuid=constants.UUID, fname=cur_fname)

    del model, mean_test_loss


def convergence_check(validation_losses):
    if len(validation_losses) <= 1:
        return False

    val_diff = validation_losses[-1] - validation_losses[-2]
    return val_diff >= 0.


def fit_model_to_data(logger, constants, model_class, params_model, exp_num):
    node_indices, spike_times, spike_indices = data_util.load_sparse_data(constants.data_path)
    params_model['N'] = len(node_indices)

    assert constants.train_iters * constants.rows_per_train_iter <= spike_times[-1], \
        "should have enough rows. desired: {}, spikes_times[-1]: {}".format(
            constants.train_iters * constants.rows_per_train_iter, spike_times[-1])

    model = model_class(device=device, parameters=params_model)
    logger.log([model_class.__name__], 'initial model parameters: {}'.format(params_model))
    current_rate = torch.tensor(constants.initial_poisson_rate)  # * torch.rand((1,))[0]
    parameters = {}
    for p_i, param in enumerate(list(model.parameters())):
        parameters[p_i] = [param.clone().detach().numpy()]
    # parameters[p_i + 1] = [current_rate.clone().detach().numpy()]

    optim = constants.optimiser(list(model.parameters()), lr=constants.learn_rate)

    prev_state_dict = model.state_dict()
    train_losses = []; validation_losses = []; prev_spike_index = 0; train_i = 0; converged = False
    while not (converged and (train_i < constants.train_iters)):
        logger.log([ExperimentType.DataDriven], 'training iteration #{}'.format(train_i))
        prev_spike_index, targets = data_util.get_spike_train_matrix(index_last_step=prev_spike_index,
                                                                     advance_by_t_steps=constants.rows_per_train_iter,
                                                                     spike_times=spike_times, spike_indices=spike_indices,
                                                                     node_numbers=node_indices)

        prev_state_dict = model.state_dict()
        avg_train_loss = fit_mini_batches(model, inputs=None, target_spiketrain=targets, current_rate=current_rate,
                                          optimiser=optim, constants=constants, train_i=train_i, logger=logger)
        logger.log(['avg train loss', avg_train_loss])
        train_losses.append(avg_train_loss)

        # gradients = []
        for param_i, param in enumerate(list(model.parameters())):
            logger.log('-', 'parameter #{}: {}'.format(param_i, param))
            logger.log('-', 'parameter #{} gradient: {}'.format(param_i, param.grad))
            parameters[param_i].append(param.clone().detach().numpy())
            # gradients.append(param.grad)
        # parameters[param_i + 1].append(current_rate.clone().detach().numpy())
        # gradients.append(current_rate.grad)

        # converged = convergence_check(gradients)

        model.reset_hidden_state()
        current_rate = current_rate.clone().detach()

        # if train_i % constants.evaluate_step == 0 or (converged or (train_i+1 >= constants.train_iters)):
        prev_spike_index, targets = data_util.get_spike_train_matrix(index_last_step=prev_spike_index,
                                                                     advance_by_t_steps=constants.rows_per_train_iter,
                                                                     spike_times=spike_times,
                                                                     spike_indices=spike_indices,
                                                                     node_numbers=node_indices)
        validation_inputs = poisson_input(rate=current_rate, t=constants.rows_per_train_iter, N=model.N)
        validation_loss = evaluate_loss(model, inputs=validation_inputs, target_spiketrain=targets, uuid=constants.UUID,
                                        tau_van_rossum=constants.tau_van_rossum, label='train i: {}'.format(train_i),
                                        exp_type=ExperimentType.DataDriven, train_i=train_i, exp_num=exp_num, constants=constants)
        logger.log(['validation loss', validation_loss], '')
        validation_losses.append(validation_loss)
        converged = convergence_check(validation_losses)

        model.reset_hidden_state()
        current_rate = current_rate.clone().detach()

    model.load_state_dict(prev_state_dict)
    stats_training_iterations(parameters, model, train_losses, validation_losses, constants, logger, ExperimentType.DataDriven.name,
                              target_parameters=False, exp_num=exp_num, train_i=train_i)

    test_loss_detached = torch.tensor(validation_losses).clone().detach()
    train_loss_detached = torch.tensor(train_losses).clone().detach()
    del targets, model, train_losses, validation_losses  # cleanup

    return parameters, train_loss_detached, test_loss_detached, train_i


def run_exp_loop(logger, constants, model_class, free_model_parameters):
    all_recovered_params = {}
    target_parameters = False
    for exp_i in range(constants.N_exp):
        torch.manual_seed(exp_i)
        np.random.seed(exp_i)

        params_model = randomise_parameters(free_model_parameters, coeff=torch.tensor(0.1))

        recovered_parameters, train_losses, test_losses, train_i = \
            fit_model_to_data(logger, constants, model_class, params_model, exp_num=exp_i)

        if train_i >= constants.train_iters:
            # TODO: Sanity check firing rate?
            print('DID NOT CONVERGE FOR SEED, CONTINUING ON TO NEXT SEED. exp_i: {}, train_i: {}, train_losses: {}, test_losses: {}'
                  .format(exp_i, train_i, train_losses, test_losses))

        for p_i, p in enumerate(recovered_parameters.values()):
            if exp_i == 0:
                all_recovered_params[p_i] = [p]
            else:
                all_recovered_params[p_i].append(p)

        # Note: Test losses, avg. firing rates?, ++

    plot_all_param_pairs_with_variance(all_recovered_params,
                                       uuid=constants.UUID,
                                       exp_type=ExperimentType.RetrieveFitted.name,
                                       target_params=target_parameters,
                                       param_names=model_class.parameter_names,
                                       custom_title="Average inferred parameters across experiments [{}, {}]".format(
                                           model_class.__name__, constants.optimiser),
                                       logger=logger, fname='all_inferred_params_{}'.format(model_class.__name__))


def start_exp(constants, model_class):
    logger = Log.Logger(ExperimentType.RetrieveFitted, constants, prefix=model_class.__name__)
    logger.log([constants.__str__()], 'Starting exp. with the listed hyperparameters.')

    if model_class in [LIF, LIF_R, LIF_ASC, LIF_R_ASC, GLIF]:
        free_parameters = {'w_mean': 0.3, 'w_var': 0.5, 'C_m': 1.5, 'G': 0.8, 'R_I': 18., 'E_L': -60.,
                           'delta_theta_s': 25., 'b_s': 0.4, 'f_v': 0.14, 'delta_V': 12., 'f_I': 0.4, 'I_A': 1.,
                           'b_v': 0.5, 'a_v': 0.5, 'theta_inf': -25.}
    else:
        logger.log([], 'Model class not supported.')
        sys.exit(1)

    run_exp_loop(logger, constants, model_class, free_parameters)
