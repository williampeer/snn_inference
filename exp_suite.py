import sys

from torch import tensor as T

import Log
import data_util
from Constants import ExperimentType
from Models.BaselineSNN import BaselineSNN
from Models.Izhikevich import Izhikevich, IzhikevichWeightsOnly, IzhikevichStable
from Models.LIF import LIF
from eval import evaluate
from experiments import *
from fit import *
from plot import *

torch.autograd.set_detect_anomaly(True)

# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = 'cpu'
verbose = True
# ---------------------------------------


def train_iter(inputs, targets, train_test_split_factor, model, current_rate, optims, constants, logger, train_i, exp_type_str, exp_num):
    train_inputs = None; test_inputs = None
    if inputs is not None:
        train_inputs, test_inputs = train_test_split(inputs, train_test_split_factor=train_test_split_factor)
    train_targets, test_targets = train_test_split(targets, train_test_split_factor=train_test_split_factor)
    del inputs, targets

    avg_train_loss = fit_mini_batches(model, inputs=None, target_spiketrain=train_targets,
                                      tau_van_rossum=T(constants.tau_van_rossum), current_rate=current_rate,
                                      batch_size=constants.batch_size, uuid=constants.UUID,
                                      optimisers=optims, loss_fn=constants.loss_fn, exp_type_str=exp_type_str,
                                      exp_num=exp_num, train_i=train_i)
    model.reset_hidden_state()

    if test_inputs is None:
        test_inputs = poisson_input(rate=current_rate, t=test_targets.shape[0], N=model.N)
    test_loss = evaluate(model, test_inputs, test_targets, T(constants.tau_van_rossum), train_i, uuid=constants.UUID,
                         exp_type=exp_type_str, train_i=train_i, exp_num=exp_num)

    logger.log(['avg train loss', avg_train_loss, 'test loss', test_loss],
               'train iteration #{}'.format(train_i))
    logger.log(list(model.parameters()), 'after train iteration #{}'.format(train_i))
    del train_inputs, test_inputs, train_targets, test_targets

    return avg_train_loss, test_loss


def stats_training_iterations(model_parameters, model, train_losses, test_losses, constants, logger, exp_type_str, target_parameters, exp_num):
    plot_all_param_pairs_with_variance(model_parameters,
                                       uuid=constants.UUID,
                                       exp_type=exp_type_str,
                                       target_params=target_parameters,
                                       custom_title='Inferred parameters across training iterations',
                                       fname='inferred_params_{}_exp_num_{}'.format(model.__class__.__name__, exp_num),
                                       logger=logger)

    plot_losses(training_loss=train_losses, test_loss=test_losses, uuid=constants.UUID, exp_type=exp_type_str,
                custom_title='Loss ({}, {}, lr={})'.format(model.__class__.__name__, constants.optimiser.__name__,
                                                           constants.learn_rate),
                fname='training_and_test_loss_exp_{}'.format(exp_num))

    logger.log('', 'train_losses: #{}'.format(train_losses))
    mean_test_loss = torch.mean(torch.tensor(test_losses)).data
    logger.log(['mean test loss: {}'.format(mean_test_loss)], 'test_losses: #{}'.format(test_losses))

    cur_fname = '{}_exp_num_{}_mean_loss_{}'.format(model.__class__.__name__, exp_num, mean_test_loss)
    IO.save(model, loss={'train_losses': train_losses, 'test_losses': test_losses}, uuid=constants.UUID,
            fname=cur_fname)

    del model, mean_test_loss


def fit_model_to_data(logger, constants, model_class, params_model, data_set='exp138', exp_type=ExperimentType.DataDriven, exp_num=None):
    data_index = data_util.exp_names.index(data_set)
    node_indices, spike_times, spike_indices, states = data_util.load_data(data_index)
    # states_per_train_iter = int(constants.rows_per_train_iter / constants.data_bin_size)

    assert constants.train_iters * constants.rows_per_train_iter <= spike_times[-1], \
        "should have enough rows. desired: {}, spikes_times[-1]: {}".format(
            constants.train_iters * constants.rows_per_train_iter, spike_times[-1])

    model = model_class(device=device, parameters=params_model)
    logger.log([model_class.__name__], 'initial model parameters: {}'.format(params_model))
    current_rate = torch.tensor(constants.initial_poisson_rate)  # * torch.rand((1,))[0]
    model_parameters = {}
    for p_i, param in enumerate(list(model.parameters())):
        model_parameters[p_i] = [param.clone().detach().numpy()]

    model_optim = constants.optimiser(list(model.parameters()), lr=constants.learn_rate)
    poisson_rates_optim = constants.optimiser([current_rate], lr=constants.learn_rate)
    optims = [model_optim, poisson_rates_optim]

    train_losses = []; test_losses = []; prev_spike_arr_index = 0
    for train_i in range(constants.train_iters):
        # model.reset_hidden_state()
        # inputs = poisson_input(rates=current_rates, t=constants.rows_per_train_iter)
        prev_spike_arr_index, targets = data_util.get_spike_array(index_last_step=prev_spike_arr_index,
                                                                  advance_by_t_steps=constants.rows_per_train_iter,
                                                                  spike_times=spike_times, spike_indices=spike_indices,
                                                                  node_numbers=node_indices)

        # -- in-place model modification --
        # avg_train_loss, test_loss = train_iter(inputs, targets, model, current_rates, constants, logger, train_i, exp_type.name)
        train_test_split_factor = 0.8
        avg_train_loss, test_loss = train_iter(None, targets, train_test_split_factor, model, current_rate, optims,
                                               constants, logger, train_i, exp_type.name, exp_num)
        model.reset_hidden_state()
        current_rate = current_rate.clone().detach()

        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)

        for param_i, param in enumerate(list(model.parameters())):
            logger.log('-', 'parameter #{}: {}'.format(param_i, param))
            logger.log('-', 'parameter #{} gradient: {}'.format(param_i, param.grad))
            model_parameters[param_i].append(param.clone().detach().numpy())

    stats_training_iterations(model_parameters, model, train_losses, test_losses, constants, logger, exp_type.name,
                              target_parameters=False, exp_num=exp_num)

    # del inputs, targets, model, train_losses, test_losses  # cleanup
    del targets, model, train_losses, test_losses  # cleanup

    return model_parameters


def recover_model_parameters(logger, constants, model_class, params_model, params_gen, exp_type=ExperimentType.Synthetic, exp_num=None):
    gen_model = model_class(device=device, parameters=params_gen)
    logger.log([model_class.__name__], 'gen model parameters: {}'.format(params_gen))
    gen_rate = torch.tensor(constants.initial_poisson_rate)  # * torch.rand((1,))[0]
    target_parameters = {}
    for param_i, param in enumerate(list(gen_model.parameters())):
        target_parameters[param_i] = [param.clone().detach().numpy()]

    model = model_class(device=device, parameters=params_model)
    if exp_type is ExperimentType.SanityCheck:
        if hasattr(model, 'w'):
            model.w = torch.nn.Parameter(gen_model.w.clone().detach().data, requires_grad=True)
    logger.log([model_class.__name__], 'initial model parameters: {}'.format(params_model))
    current_rate = torch.tensor(constants.initial_poisson_rate)  # * torch.rand((1,))[0]
    fitted_parameters = {}
    for p_i, param in enumerate(list(model.parameters())):
        fitted_parameters[p_i] = [param.clone().detach().numpy()]
    fitted_parameters[p_i + 1] = [current_rate.clone().detach().numpy()]

    model_optim = constants.optimiser(list(model.parameters()), lr=constants.learn_rate)
    poisson_rate_optim = constants.optimiser([current_rate], lr=constants.learn_rate)
    optims = [model_optim, poisson_rate_optim]

    train_losses = []; test_losses = []; prev_spike_arr_index = 0
    for train_i in range(constants.train_iters):
        # inputs = poisson_input(rates=current_rates, t=constants.rows_per_train_iter)
        gen_model.reset_hidden_state()
        targets = generate_synthetic_data(gen_model, poisson_rate=gen_rate, t=constants.rows_per_train_iter)

        # -- in-place model modification --
        train_test_split_factor = 0.8
        avg_train_loss, test_loss = train_iter(None, targets, train_test_split_factor, model, current_rate, optims, constants, logger, train_i, exp_type.name, exp_num)
        # train_iter(None, targets, model, current_rates, constants, logger, train_i, exp_type.name)
        model.reset_hidden_state()
        current_rate = current_rate.clone().detach()

        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)
        del avg_train_loss, test_loss

        for param_i, param in enumerate(list(model.parameters())):
            logger.log('-', 'parameter #{}: {}'.format(param_i, param))
            logger.log('-', 'parameter #{} gradient: {}'.format(param_i, param.grad))
            fitted_parameters[param_i].append(param.clone().detach().numpy())
        fitted_parameters[param_i+1].append(current_rate.clone().detach().numpy())

        del targets

    final_parameters = {}
    for param_i, param in enumerate(list(model.parameters())):
        logger.log('-', 'parameter #{}: {}'.format(param_i, param))
        logger.log('-', 'parameter #{} gradient: {}'.format(param_i, param.grad))
        final_parameters[param_i] = param.clone().detach().numpy()

    stats_training_iterations(fitted_parameters, model, train_losses, test_losses, constants, logger, exp_type.name,
                              target_parameters=target_parameters, exp_num=exp_num)

    # del inputs, targets, model, train_losses, test_losses  # cleanup
    del model, train_losses, test_losses  # cleanup

    return final_parameters, target_parameters
    # return model_parameters, target_parameters


def run_exp_loop(logger, constants, exp_type, model_class, params_model, params_gen):
    all_recovered_params = {}; recovered_parameters = None
    target_parameters = False
    for exp_i in range(constants.N_exp):
        if exp_type is ExperimentType.DataDriven:
            recovered_parameters = fit_model_to_data(logger, constants, model_class, params_model, exp_type=exp_type, exp_num=exp_i)
        elif exp_type in [ExperimentType.SanityCheck, ExperimentType.Synthetic]:
            if exp_type is ExperimentType.SanityCheck:
                params_model = params_gen.copy()
            recovered_parameters, target_parameters = recover_model_parameters(logger, constants, model_class, params_model, params_gen, exp_type=exp_type, exp_num=exp_i)
            # recover_model_parameters(logger, constants, model_class, params_model, params_gen, exp_type=exp_type)

    #     for p_i, p in enumerate(recovered_parameters.values()):
    #         if exp_i == 0:
    #             all_recovered_params[p_i] = [p]
    #         else:
    #             all_recovered_params[p_i].append(p)
    #
    # plot_all_param_pairs_with_variance_new(all_recovered_params,
    #                                    uuid=constants.UUID,
    #                                    exp_type=exp_type.name,
    #                                    target_params=target_parameters,
    #                                    custom_title="Average inferred parameters across experiments [{}, {}]".format(
    #                                            model_class.__name__, constants.optimiser),
    #                                    logger=logger, fname='all_inferred_params_{}'.format(model_class.__name__))


def start_exp(constants, model_class, experiment_type=ExperimentType.DataDriven):
    logger = Log.Logger(experiment_type, constants, prefix=model_class.__name__)
    logger.log([constants.__str__()], 'Starting exp. with the listed hyperparameters.')

    if model_class is LIF:
        static_init_parameters = {'N': 12, 'w_mean': 0.2, 'w_var': 0.3,
                             'pre_activation_coefficient': 2.0, 'post_activation_coefficient': 120.0}
        free_parameters = {'tau_m': 4.0, 'tau_g': 2.0, 'v_rest': -70.0}

    elif model_class in [Izhikevich, IzhikevichStable, IzhikevichWeightsOnly]:
        static_init_parameters = {'N': 12, 'w_mean': 0.1, 'w_var': 0.2, 'a': 0.1, 'b': 0.25}
        free_parameters = {'c': -62.5, 'd': 6., 'tau_g': 4.5}

    elif model_class is BaselineSNN:
        static_init_parameters = {'N': 12, 'w_mean': 0.6, 'w_var': 0.7}
        free_parameters = {}
        if experiment_type == ExperimentType.SanityCheck:
            static_init_parameters['w_var'] = 0.0

    else:
        logger.log([], 'Model class not supported.')
        sys.exit(1)

    if experiment_type == ExperimentType.Synthetic:
        free_parameters = randomise_parameters(free_parameters)
    params_gen = zip_dicts(free_parameters, static_init_parameters)
    params_model = zip_dicts(free_parameters, static_init_parameters)

    run_exp_loop(logger, constants, experiment_type, model_class, params_model, params_gen)
