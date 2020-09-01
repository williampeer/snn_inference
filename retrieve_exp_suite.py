import sys

from torch import tensor as T

import Log
from Constants import ExperimentType
from Models import SleepModelWrappers
from Models.GLIF import GLIF
from Models.LIF import LIF, LIF_complex
from Models.LIF_ASC import LIF_ASC
from Models.LIF_R import LIF_R
from Models.LIF_R_ASC import LIF_R_ASC
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


def recover_model_parameters(logger, constants, model_class, params_model, exp_num=None):
    # gen_model = torch.load(constants.fitted_model_path)['model']
    gen_model = SleepModelWrappers.lif_sleep_model()

    logger.log([model_class.__name__], 'gen model parameters: {}'.format(gen_model.parameters()))
    gen_rate = torch.tensor(constants.initial_poisson_rate)
    if model_class.__name__ == gen_model.__class__.__name__:
        target_parameters = {}
        for param_i, param in enumerate(list(gen_model.parameters())):
            target_parameters[param_i] = [param.clone().detach().numpy()]
        target_parameters[param_i + 1] = [gen_rate.clone().detach().numpy()]
    else:
        target_parameters = False

    params_model['N'] = gen_model.N
    model = model_class(device=device, parameters=params_model)
    logger.log([model_class.__name__], 'initial model parameters: {}'.format(params_model))
    current_rate = torch.tensor(constants.initial_poisson_rate)  # * torch.rand((1,))[0]
    fitted_parameters = {}
    for p_i, param in enumerate(list(model.parameters())):
        fitted_parameters[p_i] = [param.clone().detach().numpy()]
    fitted_parameters[p_i + 1] = [current_rate.clone().detach().numpy()]

    model_optim = constants.optimiser(list(model.parameters()), lr=constants.learn_rate)
    # poisson_rate_optim = constants.optimiser([current_rate], lr=constants.learn_rate)
    # optims = [model_optim, poisson_rate_optim]
    optims = [model_optim]

    train_losses = []; test_losses = []
    for train_i in range(constants.train_iters):
        gen_model.reset_hidden_state()
        targets = generate_synthetic_data(gen_model, poisson_rate=gen_rate, t=constants.rows_per_train_iter)

        avg_train_loss = fit_mini_batches(model, inputs=None, target_spiketrain=targets,
                                          tau_van_rossum=T(constants.tau_van_rossum), current_rate=current_rate,
                                          batch_size=constants.batch_size, uuid=constants.UUID,
                                          optimisers=optims, loss_fn=constants.loss_fn,
                                          exp_type_str=ExperimentType.RetrieveFitted.name,
                                          exp_num=exp_num, train_i=train_i, logger=logger)
        logger.log(['avg train loss', avg_train_loss])
        train_losses.append(avg_train_loss)
        model.reset_hidden_state()

        last_train_iter = (train_i == constants.train_iters - 1)
        if train_i % constants.evaluate_step == 0 or last_train_iter:
            gen_model.reset_hidden_state()
            targets = generate_synthetic_data(gen_model, poisson_rate=gen_rate, t=constants.rows_per_train_iter)

            test_inputs = poisson_input(rate=current_rate, t=constants.rows_per_train_iter, N=model.N)
            test_loss = evaluate_likelihood(model, inputs=test_inputs, target_spiketrain=targets, uuid=constants.UUID,
                                            tau_van_rossum=constants.tau_van_rossum, label='train i: {}'.format(train_i),
                                            exp_type=ExperimentType.RetrieveFitted, train_i=train_i, exp_num=exp_num,
                                            constants=constants)
            logger.log(['test loss', test_loss], '')
            test_losses.append(test_loss)

            model.reset_hidden_state()
            current_rate = current_rate.clone().detach()

        for param_i, param in enumerate(list(model.parameters())):
            logger.log('-', 'parameter #{}: {}'.format(param_i, param))
            logger.log('-', 'parameter #{} gradient: {}'.format(param_i, param.grad))
            fitted_parameters[param_i].append(param.clone().detach().numpy())
        fitted_parameters[param_i+1].append(current_rate.clone().detach().numpy())
        logger.log('-', 'rates: {}'.format(current_rate))
        logger.log('-', 'rates gradient: {}'.format(current_rate.grad))

    final_parameters = {}
    for param_i, param in enumerate(list(model.parameters())):
        logger.log('-', 'parameter #{}: {}'.format(param_i, param))
        logger.log('-', 'parameter #{} gradient: {}'.format(param_i, param.grad))
        final_parameters[param_i] = param.clone().detach().numpy()
    final_parameters[param_i + 1] = current_rate.clone().detach().numpy()

    for ctr in range(len(fitted_parameters)):
        print('fitted_parameters param #{}:'.format(ctr))
        for ij in range(len(fitted_parameters[ctr])):
            print(fitted_parameters[ctr][ij])

    stats_training_iterations(fitted_parameters, model, train_losses, test_losses, constants, logger, ExperimentType.RetrieveFitted.name,
                              target_parameters=target_parameters, exp_num=exp_num)

    train_loss_detached = torch.tensor(train_losses).clone().detach()
    # del model, train_losses, test_losses  # cleanup
    del model, train_losses, test_losses  # cleanup

    return final_parameters, target_parameters, train_loss_detached


def run_exp_loop(logger, constants, model_class, free_parameters):
    all_recovered_params = {}; recovered_parameters = None
    target_parameters = False
    for exp_i in range(constants.N_exp):
        torch.manual_seed(exp_i)
        np.random.seed(exp_i)

        convergence_criterion = False
        while_ctr = 0
        while not convergence_criterion:
            params_model = randomise_parameters(free_parameters, coeff=torch.tensor(0.10))

            recovered_parameters, target_parameters, train_losses = recover_model_parameters(logger, constants, model_class, params_model, exp_num=exp_i)
            convergence_criterion = train_losses[-1] < train_losses[0] * 0.85 or while_ctr >= 10
            if while_ctr >= 10:
                print('DID NOT CONVERGE FOR SEED, CONTINUING ON TO NEXT SEED. exp_i: {}, while_ctr: {}, train_losses{}'
                      .format(exp_i, while_ctr, train_losses))

        for p_i, p in enumerate(recovered_parameters.values()):
            if exp_i == 0:
                all_recovered_params[p_i] = [p]
            else:
                all_recovered_params[p_i].append(p)

        # Note: Test losses, avg. firing rates?, ++

    # TODO: Fixme
    # plot_all_param_pairs_with_variance_new(all_recovered_params,
    #                                    uuid=constants.UUID,
    #                                    exp_type=exp_type.name,
    #                                    target_params=target_parameters,
    #                                    custom_title="Average inferred parameters across experiments [{}, {}]".format(
    #                                            model_class.__name__, constants.optimiser),
    #                                    logger=logger, fname='all_inferred_params_{}'.format(model_class.__name__))


def start_exp(constants, model_class):
    logger = Log.Logger(ExperimentType.RetrieveFitted, constants, prefix=model_class.__name__)
    logger.log([constants.__str__()], 'Starting exp. with the listed hyperparameters.')

    if model_class in [LIF, LIF_complex, LIF_R, LIF_ASC, LIF_R_ASC, GLIF]:
        free_parameters = {'w_mean': 0.2, 'w_var': 0.3, 'tau_m': 1.5, 'tau_g': 4.0, 'v_rest': -60.0}

    else:
        logger.log([], 'Model class not supported.')
        sys.exit(1)

    run_exp_loop(logger, constants, model_class, free_parameters)
