from torch.nn.functional import poisson_nll_loss, kl_div

import model_util
import spike_metrics
from plot import *


def evaluate_loss(model, inputs, target_spiketrain, tau_van_rossum, uuid, label='', exp_type=None, train_i=None, exp_num=None, constants=None):
    assert (inputs.shape[0] == target_spiketrain.shape[0]), "inputs and targets should have same shape. inputs shape: {}, targets shape: {}"\
        .format(inputs.shape, target_spiketrain.shape)

    model_spiketrain = model_util.feed_inputs_sequentially_return_spiketrain(model, inputs)

    print('-- sanity-checks --')
    print('model:')
    sanity_checks(torch.round(model_spiketrain))
    print('target:')
    sanity_checks(target_spiketrain)
    print('-- sanity-checks-done --')

    loss = calculate_loss(model_spiketrain, target_spiketrain, loss_fn=constants.loss_fn, tau_vr=tau_van_rossum)
    print('loss:', loss)

    if exp_type is None:
        exp_type_str = 'default'
    else:
        exp_type_str = exp_type.name
    plot_spiketrains_side_by_side(model_spiketrain, target_spiketrain, uuid=uuid,
                                  exp_type=exp_type_str, title='Spiketrains test set ({}, loss: {:.3f})'.format(label, loss),
                                  fname='spiketrains_test_set_{}_exp_{}_train_iter_{}'.format(model.__class__.__name__, exp_num, train_i))
    np_loss = loss.clone().detach().numpy()
    loss = None
    return np_loss


def calculate_loss(output, target, loss_fn, tau_vr=None):
    if loss_fn.__contains__('van_rossum_dist'):
        loss = spike_metrics.van_rossum_dist(output, target, tau_vr)
    elif loss_fn.__contains__('poisson_nll'):
        loss = poisson_nll_loss(output, target)
    elif loss_fn.__contains__('kl_div'):
        loss = kl_div(output, target)
    elif loss_fn.__contains__('van_rossum_squared'):
        loss = spike_metrics.van_rossum_squared_distance(output, target, tau_vr)
    elif loss_fn.__contains__('mse'):
        loss = spike_metrics.mse(output, target)
    elif loss_fn.__contains__('firing_rate_distance'):
        loss = spike_metrics.firing_rate_distance(output, target)
    else:
        raise NotImplementedError("Loss function not supported.")

    # return loss
    return loss + spike_metrics.firing_rate_distance(output, target)  # add term for firing rate.

# --------------------------------------------------------


def sanity_checks(spiketrain):
    neuron_spikes = spiketrain.sum(0)
    silent_neurons = (neuron_spikes == 0).sum()

    print('# silent neurons: ', silent_neurons)
    print('spikes per neuron:', neuron_spikes)
