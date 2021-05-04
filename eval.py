from torch.nn.functional import kl_div
from enum import Enum

import model_util
import spike_metrics
from experiments import poisson_input, release_computational_graph
from plot import *


def evaluate_loss(model, inputs, p_rate, target_spiketrain, label='', exp_type=None, train_i=None, exp_num=None, constants=None, converged=False):
    if inputs is not None:
        assert (inputs.shape[0] == target_spiketrain.shape[0]), \
            "inputs and targets should have same shape. inputs shape: {}, targets shape: {}".format(inputs.shape, target_spiketrain.shape)
    else:
        inputs = poisson_input(rate=p_rate, t=target_spiketrain.shape[0], N=model.N)

    model_spike_train = model_util.feed_inputs_sequentially_return_spike_train(model, inputs)

    print('-- sanity-checks --')
    print('model:')
    sanity_checks(torch.round(model_spike_train))
    print('target:')
    sanity_checks(target_spiketrain)
    print('-- sanity-checks-done --')

    loss = calculate_loss(model_spike_train, target_spiketrain, loss_fn=constants.loss_fn, N=model.N,
                          tau_vr=constants.tau_van_rossum)
    print('loss:', loss)

    if exp_type is None:
        exp_type_str = 'default'
    else:
        exp_type_str = exp_type.name

    if train_i % constants.evaluate_step == 0 or converged or train_i == constants.train_iters -1:
        plot_spiketrains_side_by_side(model_spike_train, target_spiketrain, uuid=constants.UUID, exp_type=exp_type_str,
                                      title='Spike trains test set ({}, loss: {:.3f})'.format(label, loss),
                                      fname='spiketrains_test_set_{}_exp_{}_train_iter_{}'.format(model.__class__.__name__, exp_num, train_i))
    np_loss = loss.clone().detach().numpy()
    release_computational_graph(model, p_rate, inputs)
    loss = None
    return np_loss


class LossFn(Enum):
    FIRING_RATE_DIST = 'frd'
    VAN_ROSSUM_DIST = 'vrd'
    MSE = 'mse'
    KL_DIV = 'kl_div'


def calculate_loss(output, target, loss_fn, N, tau_vr=None, train_f=0.):
    lfn = LossFn[loss_fn]
    if lfn == LossFn.KL_DIV:
        loss = kl_div(output, target)
    elif lfn == LossFn.MSE:
        loss = spike_metrics.mse(output, target)
    elif lfn == LossFn.VAN_ROSSUM_DIST:
        loss = spike_metrics.van_rossum_dist(output, target, tau_vr)
    elif lfn == LossFn.FIRING_RATE_DIST:
        loss = spike_metrics.firing_rate_distance(output, target)
    else:
        raise NotImplementedError("Loss function not supported.")

    silent_penalty = spike_metrics.silent_penalty_term(output, target)
    # activity_term = 0.5 * spike_metrics.normalised_overall_activity_term(output)  # TEST
    # return loss + silent_penalty + activity_term
    return loss + silent_penalty  # TEST
    # return loss

# --------------------------------------------------------


def sanity_checks(spiketrain):
    neuron_spikes = spiketrain.sum(0)
    silent_neurons = (neuron_spikes == 0).sum()

    print('# silent neurons: ', silent_neurons)
    print('spikes per neuron:', neuron_spikes)
