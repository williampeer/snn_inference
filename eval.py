from torch.nn.functional import poisson_nll_loss

import model_util
import spike_metrics
from plot import *


def evaluate_likelihood(model, inputs, target_spiketrain, tau_van_rossum, uuid, label='', exp_type=None, train_i=None, exp_num=None, constants=None):
    assert (inputs.shape[0] == target_spiketrain.shape[0]), "inputs and targets should have same shape. inputs shape: {}, targets shape: {}"\
        .format(inputs.shape, target_spiketrain.shape)

    membrane_potentials, model_spiketrain = model_util.feed_inputs_sequentially_return_spikes_and_potentials(model, inputs)

    print('-- sanity-checks --')
    print('model:')
    sanity_checks(torch.round(model_spiketrain))
    print('target:')
    sanity_checks(target_spiketrain)
    print('-- sanity-checks-done --')

    if constants.loss_fn.__contains__('van_rossum_dist'):
        loss = float(spike_metrics.van_rossum_dist(model_spiketrain, target_spiketrain, tau=tau_van_rossum).detach().data)
    elif constants.loss_fn.__contains__('poisson_nll'):
        loss = float(poisson_nll_loss(model_spiketrain, target_spiketrain).detach().data)
    elif constants.loss_fn.__contains__('van_rossum_squared'):
        loss = float(spike_metrics.van_rossum_squared_distance(model_spiketrain, target_spiketrain, tau=tau_van_rossum).detach().data)
    elif constants.loss_fn.__contains__('mse'):
        loss = float(spike_metrics.mse(model_spiketrain, target_spiketrain).detach().data)
    print('loss:', loss)

    if exp_type is None:
        exp_type_str = 'default'
    else:
        exp_type_str = exp_type.name
    plot_spiketrains_side_by_side(model_spiketrain, target_spiketrain, uuid=uuid,
                                  exp_type=exp_type_str, title='Spiketrains test set ({}, loss: {:.3f})'.format(label, loss),
                                  fname='spiketrains_test_set_{}_exp_{}_train_iter_{}'.format(model.__class__.__name__, exp_num, train_i))

    return loss

# --------------------------------------------------------


def sanity_checks(spiketrain):
    neuron_spikes = spiketrain.sum(0)
    silent_neurons = (neuron_spikes == 0).sum()

    print('# silent neurons: ', silent_neurons)
    print('spikes per neuron:', neuron_spikes)


# def eval_parameters_1d(recovered_params_list, target_params, uuid, custom_title=False):
#     assert len(recovered_params_list) == len(target_params), \
#         "p.1: {}, p.2: {}".format(len(recovered_params_list[0]), len(target_params))
#
#     for param_i, parameters in enumerate(recovered_params_list.values()):
#         plot_fitted_vs_target_parameters(parameters, target_params[param_i].data, uuid, custom_title=custom_title)
