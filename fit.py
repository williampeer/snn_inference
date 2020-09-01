import torch
from torch.nn.functional import poisson_nll_loss

import model_util
import spike_metrics
from experiments import poisson_input


def fit_mini_batches(model, inputs, target_spiketrain, tau_van_rossum, current_rate, batch_size, uuid,
                     optimisers, loss_fn='van_rossum_dist', exp_type_str='default', exp_num=None, train_i=None, logger=None):
    if inputs is not None:
        assert inputs.shape[0] == target_spiketrain.shape[0], \
            "inputs shape: {}, target spiketrain shape: {}".format(inputs.shape, target_spiketrain.shape)

    batch_N = int(target_spiketrain.shape[0]/batch_size)
    assert batch_N > 0, "batch_N was not above zero. batch_N: {}".format(batch_N)
    print('num. of batches of size {}: {}'.format(batch_size, batch_N))
    batch_losses = []
    loss = None; cur_inputs = None; loss_per_node = False
    for batch_i in range(batch_N):
        print('batch #{}'.format(batch_i))

        model.reset_hidden_state()
        # current_rate = current_rate.clone().detach()
        # current_rate = torch.abs(current_rate.clone().detach())

        if inputs is not None:
            spikes = model_util.feed_inputs_sequentially_return_spiketrain(model, inputs[batch_size*batch_i:batch_size*(batch_i+1)])
        else:
            cur_inputs = poisson_input(rate=current_rate, t=batch_size, N=model.N)
            spikes = model_util.feed_inputs_sequentially_return_spiketrain(model, cur_inputs)

        if loss_fn.__contains__('van_rossum_dist'):
            loss = spike_metrics.van_rossum_dist(spikes, target_spiketrain[batch_size * batch_i:batch_size * (batch_i + 1)].detach(), tau=tau_van_rossum)
        elif loss_fn.__contains__('poisson_nll'):
            loss = poisson_nll_loss(spikes, target_spiketrain[batch_size * batch_i:batch_size * (batch_i + 1)].detach())
        elif loss_fn.__contains__('van_rossum_squared'):
            loss = spike_metrics.van_rossum_squared_distance(spikes, target_spiketrain[batch_size*batch_i:batch_size*(batch_i+1)].detach(), tau=tau_van_rossum)
        elif loss_fn.__contains__('mse'):
            loss = spike_metrics.mse(spikes, target_spiketrain[batch_size*batch_i:batch_size*(batch_i+1)].detach())
        else:
            raise NotImplementedError("Loss function not supported.")

        sut_tar_sum = target_spiketrain[batch_size*batch_i:batch_size*(batch_i+1)].sum()
        print('DEBUG. # target spikes in batch: {}'.format(sut_tar_sum))
        print('batch loss: {}'.format(loss))

        for optim in optimisers:
            optim.zero_grad()

        loss.backward(retain_graph=True)
        batch_losses.append(loss.clone().detach().data)

        for optim in optimisers:
            # print('stepping with optimiser: {}'.format(optim))
            optim.step()
            print('DEBUG: Stepping in optim.')

    # plot_losses_nodes(batch_losses, uuid, exp_type_str, 'Batch loss')
    avg_batch_loss = torch.mean(torch.tensor(batch_losses))

    logger.log({'train_i': train_i}, 'avg_batch_loss: {}'.format(avg_batch_loss))
    del loss, spikes, inputs, cur_inputs, batch_losses

    return float(avg_batch_loss.clone().detach().data)
