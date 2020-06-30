import sys

from torch import tensor as T

import Log
import data_util
from Constants import ExperimentType
from Models.BaselineSNN import BaselineSNN
from Models.Izhikevich import Izhikevich, IzhikevichWeightsOnly, IzhikevichStable
from Models.LIF import LIF
from eval import evaluate_likelihood
from experiments import *
from fit import *
from plot import *

torch.autograd.set_detect_anomaly(True)

# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = 'cpu'
verbose = True
# ---------------------------------------


def stats_training_iterations(model_parameters, model, train_losses, test_losses, constants, logger, exp_type_str, target_parameters, exp_num):
    plot_all_param_pairs_with_variance(model_parameters,
                                       uuid=constants.UUID,
                                       exp_type=exp_type_str,
                                       target_params=target_parameters,
                                       custom_title='Inferred parameters across training iterations',
                                       fname='inferred_params_{}_exp_num_{}'.format(model.__class__.__name__, exp_num),
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


def fit_model_to_data(logger, constants, model_class, params_model, data_set='exp138', exp_type=ExperimentType.DataDriven, exp_num=None):
    data_index = data_util.exp_names.index(data_set)
    node_indices, spike_times, spike_indices, states = data_util.load_data(data_index)
    params_model['N'] = len(node_indices)

    # states_per_train_iter = int(constants.rows_per_train_iter / constants.data_bin_size)

    assert constants.train_iters * constants.rows_per_train_iter <= spike_times[-1], \
        "should have enough rows. desired: {}, spikes_times[-1]: {}".format(
            constants.train_iters * constants.rows_per_train_iter, spike_times[-1])

    model = model_class(device=device, parameters=params_model)
    logger.log([model_class.__name__], 'initial model parameters: {}'.format(params_model))
    current_rate = torch.tensor(constants.initial_poisson_rate)  # * torch.rand((1,))[0]
    parameters = {}
    for p_i, param in enumerate(list(model.parameters())):
        parameters[p_i] = [param.clone().detach().numpy()]
    parameters[p_i + 1] = [current_rate.clone().detach().numpy()]

    model_optim = constants.optimiser(list(model.parameters()), lr=constants.learn_rate)
    poisson_rates_optim = constants.optimiser([current_rate], lr=constants.learn_rate)
    optims = [model_optim, poisson_rates_optim]

    train_losses = []; test_losses = []; prev_spike_arr_index = 0
    for train_i in range(constants.train_iters):
        logger.log([exp_type], 'training iteration #{}'.format(train_i))
        prev_spike_arr_index, targets = data_util.get_spike_array(index_last_step=prev_spike_arr_index,
                                                                  advance_by_t_steps=constants.rows_per_train_iter,
                                                                  spike_times=spike_times, spike_indices=spike_indices,
                                                                  node_numbers=node_indices)

        avg_train_loss = fit_mini_batches(model, inputs=None, target_spiketrain=targets,
                                          tau_van_rossum=T(constants.tau_van_rossum), current_rate=current_rate,
                                          batch_size=constants.batch_size, uuid=constants.UUID,
                                          optimisers=optims, loss_fn=constants.loss_fn, exp_type_str=exp_type.name,
                                          exp_num=exp_num, train_i=train_i, logger=logger)
        logger.log(['avg train loss', avg_train_loss])
        train_losses.append(avg_train_loss)
        model.reset_hidden_state()
        current_rate = current_rate.clone().detach()

        last_train_iter = (train_i == constants.train_iters-1)
        if train_i % constants.evaluate_step == 0 or last_train_iter:
            prev_spike_arr_index, targets = data_util.get_spike_array(index_last_step=prev_spike_arr_index,
                                                                      advance_by_t_steps=constants.rows_per_train_iter,
                                                                      spike_times=spike_times,
                                                                      spike_indices=spike_indices,
                                                                      node_numbers=node_indices)
            test_inputs = poisson_input(rate=current_rate, t=constants.rows_per_train_iter, N=model.N)

            test_loss = evaluate_likelihood(model, inputs=test_inputs, target_spiketrain=targets, uuid=constants.UUID,
                                            tau_van_rossum=constants.tau_van_rossum, label='train i: {}'.format(train_i),
                                            exp_type=exp_type, train_i=train_i, exp_num=exp_num)
            logger.log(['test loss', test_loss], '')
            test_losses.append(test_loss)

            model.reset_hidden_state()
            current_rate = current_rate.clone().detach()

        for param_i, param in enumerate(list(model.parameters())):
            logger.log('-', 'parameter #{}: {}'.format(param_i, param))
            logger.log('-', 'parameter #{} gradient: {}'.format(param_i, param.grad))
            parameters[param_i].append(param.clone().detach().numpy())
        parameters[param_i + 1].append(current_rate.clone().detach().numpy())

    stats_training_iterations(parameters, model, train_losses, test_losses, constants, logger, exp_type.name, target_parameters=False, exp_num=exp_num)

    del targets, model, train_losses, test_losses  # cleanup

    return parameters


def recover_model_parameters(logger, constants, model_class, params_model, params_gen, exp_type=ExperimentType.Synthetic, exp_num=None):
    if exp_type is ExperimentType.RetrieveFitted:
        gen_model = torch.load(constants.fitted_model_path)['model']
    else:
        gen_model = model_class(device=device, parameters=params_gen)

    logger.log([model_class.__name__], 'gen model parameters: {}'.format(params_gen))
    gen_rate = torch.tensor(constants.initial_poisson_rate)  # * torch.rand((1,))[0]
    target_parameters = {}
    for param_i, param in enumerate(list(gen_model.parameters())):
        target_parameters[param_i] = [param.clone().detach().numpy()]

    params_model['N'] = gen_model.N
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

    train_losses = []; test_losses = []
    for train_i in range(constants.train_iters):
        gen_model.reset_hidden_state()
        targets = generate_synthetic_data(gen_model, poisson_rate=gen_rate, t=constants.rows_per_train_iter)

        avg_train_loss = fit_mini_batches(model, inputs=None, target_spiketrain=targets,
                                          tau_van_rossum=T(constants.tau_van_rossum), current_rate=current_rate,
                                          batch_size=constants.batch_size, uuid=constants.UUID,
                                          optimisers=optims, loss_fn=constants.loss_fn, exp_type_str=exp_type.name,
                                          exp_num=exp_num, train_i=train_i, logger=logger)
        logger.log(['avg train loss', avg_train_loss])
        train_losses.append(avg_train_loss)
        model.reset_hidden_state()
        current_rate = current_rate.clone().detach()

        last_train_iter = (train_i == constants.train_iters - 1)
        if train_i % constants.evaluate_step == 0 or last_train_iter:
            gen_model.reset_hidden_state()
            targets = generate_synthetic_data(gen_model, poisson_rate=gen_rate, t=constants.rows_per_train_iter)

            test_inputs = poisson_input(rate=current_rate, t=constants.rows_per_train_iter, N=model.N)
            test_loss = evaluate_likelihood(model, inputs=test_inputs, target_spiketrain=targets, uuid=constants.UUID,
                                            tau_van_rossum=constants.tau_van_rossum, label='train i: {}'.format(train_i),
                                            exp_type=exp_type, train_i=train_i, exp_num=exp_num)
            logger.log(['test loss', test_loss], '')
            test_losses.append(test_loss)

            model.reset_hidden_state()
            current_rate = current_rate.clone().detach()

        for param_i, param in enumerate(list(model.parameters())):
            logger.log('-', 'parameter #{}: {}'.format(param_i, param))
            logger.log('-', 'parameter #{} gradient: {}'.format(param_i, param.grad))
            fitted_parameters[param_i].append(param.clone().detach().numpy())
        fitted_parameters[param_i+1].append(current_rate.clone().detach().numpy())

    final_parameters = {}
    for param_i, param in enumerate(list(model.parameters())):
        logger.log('-', 'parameter #{}: {}'.format(param_i, param))
        logger.log('-', 'parameter #{} gradient: {}'.format(param_i, param.grad))
        final_parameters[param_i] = param.clone().detach().numpy()
    fitted_parameters[param_i + 1].append(current_rate.clone().detach().numpy())

    stats_training_iterations(fitted_parameters, model, train_losses, test_losses, constants, logger, exp_type.name,
                              target_parameters=target_parameters, exp_num=exp_num)

    # del model, train_losses, test_losses  # cleanup
    del model, train_losses, test_losses  # cleanup

    return final_parameters, target_parameters
    # return model_parameters, target_parameters


def run_exp_loop(logger, constants, exp_type, model_class, params_model, params_gen):
    all_recovered_params = {}; recovered_parameters = None
    target_parameters = False
    for exp_i in range(constants.N_exp):
        if exp_type is ExperimentType.DataDriven:
            recovered_parameters = fit_model_to_data(logger, constants, model_class, params_model,
                                                     data_set=constants.data_set, exp_type=exp_type, exp_num=exp_i)
        elif exp_type in [ExperimentType.SanityCheck, ExperimentType.Synthetic, ExperimentType.RetrieveFitted]:
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
        static_init_parameters = {'N': 12, 'w_mean': 0.1, 'w_var': 0.3, 'pre_activation_coefficient': 2.0,
                             'post_activation_coefficient': 100.0}
        free_parameters = {'tau_m': 2.0, 'tau_g': 2.0, 'v_rest': -60.0}

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

    if experiment_type in [ExperimentType.Synthetic]:
        params_gen = zip_dicts(randomise_parameters(free_parameters, torch.tensor(0.5)), static_init_parameters).copy()
        params_model = zip_dicts(randomise_parameters(free_parameters, torch.tensor(0.5)), static_init_parameters).copy()
    else:
        params_gen = zip_dicts(free_parameters, static_init_parameters).copy()
        params_model = zip_dicts(free_parameters, static_init_parameters).copy()

    run_exp_loop(logger, constants, experiment_type, model_class, params_model, params_gen)
