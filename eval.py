from torch.nn.functional import poisson_nll_loss, kl_div

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

    model_spiketrain = model_util.feed_inputs_sequentially_return_spiketrain(model, inputs)

    print('-- sanity-checks --')
    print('model:')
    sanity_checks(torch.round(model_spiketrain))
    print('target:')
    sanity_checks(target_spiketrain)
    print('-- sanity-checks-done --')

    loss = calculate_loss(model_spiketrain, target_spiketrain, loss_fn=constants.loss_fn, N=model.N,
                          tau_vr=constants.tau_van_rossum)
    print('loss:', loss)

    if exp_type is None:
        exp_type_str = 'default'
    else:
        exp_type_str = exp_type.name

    if train_i % constants.evaluate_step == 0 or converged or train_i == constants.train_iters -1:
        plot_spiketrains_side_by_side(model_spiketrain, target_spiketrain, uuid=constants.UUID, exp_type=exp_type_str,
                                      title='Spike trains test set ({}, loss: {:.3f})'.format(label, loss),
                                      fname='spiketrains_test_set_{}_exp_{}_train_iter_{}'.format(model.__class__.__name__, exp_num, train_i))
    np_loss = loss.clone().detach().numpy()
    release_computational_graph(model, p_rate, inputs)
    loss = None
    return np_loss


def calculate_loss(output, target, loss_fn, N, tau_vr=None):
    if loss_fn.__contains__('vrd'):
        loss = spike_metrics.van_rossum_dist(output, target, tau_vr)
    elif loss_fn.__contains__('vrdts'):
        loss = spike_metrics.van_rossum_dist_two_sided(output, target, tau_vr)
    elif loss_fn.__contains__('poisson_nll'):
        loss = poisson_nll_loss(output, target)
    elif loss_fn.__contains__('kl_div'):
        loss = kl_div(output, target)
    elif loss_fn.__contains__('van_rossum_squared'):
        loss = spike_metrics.van_rossum_squared_distance(output, target, tau_vr)
    elif loss_fn.__contains__('mse'):
        loss = spike_metrics.mse(output, target)
    elif loss_fn.__contains__('frd'):
        loss = spike_metrics.firing_rate_distance(output, target)
    elif loss_fn.__contains__('kldfrd'):
        kld_loss = kl_div(output, target)
        frd_loss = 0.5 * spike_metrics.firing_rate_distance(output, target)  # add term for firing rate.
        loss = kld_loss + frd_loss
    elif loss_fn.__contains__('pnllfrd'):
        pnll_loss = poisson_nll_loss(output, target, tau_vr)
        frd_loss = 0.5 * spike_metrics.firing_rate_distance(output, target)  # add term for firing rate.
        loss = pnll_loss + frd_loss
    elif loss_fn.__contains__('vrdtsfrd'):
        vrdts_loss = spike_metrics.van_rossum_dist_two_sided(output, target, tau_vr)
        frd_loss = spike_metrics.firing_rate_distance(output, target)  # add term for firing rate.
        loss = vrdts_loss + frd_loss
    elif loss_fn.__contains__('vrdfrd'):
        vrd_loss = spike_metrics.van_rossum_dist(output, target, tau_vr)
        frd_loss = spike_metrics.firing_rate_distance(output, target)  # add term for firing rate.
        loss = vrd_loss + frd_loss
    elif loss_fn.__contains__('vrdsp'):
        vrd_loss = spike_metrics.van_rossum_dist(output, target, tau_vr)
        silent_penalty_term = spike_metrics.silent_penalty_term(output, target)
        loss = vrd_loss + silent_penalty_term
    elif loss_fn.__contains__('free_label_vr'):
        loss = spike_metrics.greedy_shortest_dist_vr(spikes=output, target_spikes=target, tau=tau_vr)
    elif loss_fn.__contains__('free_label_rate_dist'):
        loss = spike_metrics.shortest_dist_rates(spikes=output, target_spikes=target)
    elif loss_fn.__contains__('free_label_rate_dist_w_penalty'):
        loss = spike_metrics.shortest_dist_rates_w_silent_penalty(spikes=output, target_spikes=target)

    else:
        raise NotImplementedError("Loss function not supported.")

    return loss

# --------------------------------------------------------


def sanity_checks(spiketrain):
    neuron_spikes = spiketrain.sum(0)
    silent_neurons = (neuron_spikes == 0).sum()

    print('# silent neurons: ', silent_neurons)
    print('spikes per neuron:', neuron_spikes)
