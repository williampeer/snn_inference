import torch

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
    batch_losses = model.N * [[]]
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

        if loss_fn.__contains__('van_rossum_dist_per_node'):
            loss = spike_metrics.van_rossum_dist_per_node(spikes, target_spiketrain[batch_size * batch_i:batch_size * (batch_i + 1)].detach(), tau=tau_van_rossum)
            assert len(loss) == model.N, "loss: {}".format(loss)
            loss_per_node = True
        elif loss_fn.__contains__('van_rossum_squared_per_node'):
            loss = spike_metrics.van_rossum_squared_per_node(spikes, target_spiketrain[batch_size * batch_i:batch_size * (batch_i + 1)].detach(), tau=tau_van_rossum)
            assert len(loss) == model.N, "loss: {}".format(loss)
            loss_per_node = True
        elif loss_fn.__contains__('mse_per_node'):
            loss = spike_metrics.mse_per_node(spikes, target_spiketrain[batch_size * batch_i:batch_size * (batch_i + 1)].detach())
            assert len(loss) == model.N, "loss: {}".format(loss)
            loss_per_node = True
        elif loss_fn.__contains__('van_rossum_dist'):
            loss = spike_metrics.van_rossum_dist(spikes, target_spiketrain[batch_size * batch_i:batch_size * (batch_i + 1)].detach(), tau=tau_van_rossum)
        elif loss_fn.__contains__('van_rossum_squared'):
            loss = spike_metrics.van_rossum_squared_distance(spikes, target_spiketrain[batch_size*batch_i:batch_size*(batch_i+1)].detach(), tau=tau_van_rossum)
        elif loss_fn.__contains__('mse'):
            loss = spike_metrics.mse(spikes, target_spiketrain[batch_size*batch_i:batch_size*(batch_i+1)].detach())
        else:
            raise NotImplementedError("Loss function not supported.")

        print('batch loss: {}'.format(loss))

        for optim in optimisers:
            optim.zero_grad()

        if loss_per_node:
            for l_i in range(len(loss)):
                loss[l_i].backward(retain_graph=True)
                batch_losses[l_i].append(loss[l_i].clone().detach().data)
        else:
            loss.backward(retain_graph=True)
            batch_losses[0].append(loss.clone().detach().data)
        # print('batch_losses: {}'.format(batch_losses))

        for optim in optimisers:
            # print('stepping with optimiser: {}'.format(optim))
            optim.step()

    if not loss_per_node:
        batch_losses = [batch_losses[0]]
    # plot_losses_nodes(batch_losses, uuid, exp_type_str, 'Batch loss')

    if loss_per_node:
        avg_batch_loss = []
        for b_losses in batch_losses:
            avg_batch_loss.append(torch.mean(torch.tensor(b_losses)))
        avg_batch_loss = torch.mean(torch.tensor(avg_batch_loss))
    else:
        avg_batch_loss = torch.mean(torch.tensor(batch_losses[0]))

    logger.log({'train_i': train_i}, 'avg_batch_loss: {}'.format(avg_batch_loss))
    del loss, spikes, inputs, cur_inputs, batch_losses

    return float(avg_batch_loss.clone().detach().data)
