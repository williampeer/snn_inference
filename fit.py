import numpy as np
import torch

import model_util
from Constants import ExperimentType
from Models.GLIF import GLIF
from Models.LIF import LIF
from eval import calculate_loss
from experiments import poisson_input, release_computational_graph


def fit_mini_batches(model, gen_inputs, target_spiketrain, poisson_input_rate, optimiser, constants, train_i=None, logger=None):
    if gen_inputs is not None:
        assert gen_inputs.shape[0] == target_spiketrain.shape[0], \
            "inputs shape: {}, target spiketrain shape: {}".format(gen_inputs.shape, target_spiketrain.shape)

    tau_vr = torch.tensor(constants.tau_van_rossum)
    batch_size = constants.batch_size
    batch_N = int(target_spiketrain.shape[0]/batch_size)
    assert batch_N > 0, "batch_N was not above zero. batch_N: {}".format(batch_N)
    print('num. of batches of size {}: {}'.format(batch_size, batch_N))
    batch_losses = []; avg_abs_grads = []
    for _ in range(len(list(model.parameters()))+1):
        avg_abs_grads.append([])

    optimiser.zero_grad()
    poisson_input_rate.grad = torch.tensor(0.)
    for batch_i in range(batch_N):
        print('batch #{}'.format(batch_i))

        if constants.EXP_TYPE is ExperimentType.SanityCheck and gen_inputs is not None:
            current_inputs = gen_inputs[batch_size * batch_i:batch_size * (batch_i + 1)]
            current_inputs.retain_grad()
        else:
            current_inputs = poisson_input(rate=poisson_input_rate, t=batch_size, N=model.N)
            current_inputs.retain_grad()
        spikes = model_util.feed_inputs_sequentially_return_spiketrain(model, current_inputs)

        loss = calculate_loss(spikes, target_spiketrain[batch_size * batch_i:batch_size * (batch_i + 1)].detach(),
                              loss_fn=constants.loss_fn, N = model.N, tau_vr = tau_vr)

        loss.backward(retain_graph=True)
        poisson_input_rate.grad += torch.mean(current_inputs.grad)

        optimiser.step()

        # print('list(model.parameters())', list(model.parameters()))
        for p_i, param in enumerate(list(model.parameters())):
            # print('p_i, param.grad', p_i, param.grad)
            avg_abs_grads[p_i].append(np.mean(np.abs(param.grad.clone().detach().numpy())))
        avg_abs_grads[p_i + 1].append(np.abs(poisson_input_rate.grad.clone().detach().numpy()))
        # print('p_i+1, poisson_input_rate.grad', p_i + 1, poisson_input_rate.grad)

        print('batch loss: {}'.format(loss))
        batch_losses.append(float(loss.clone().detach().data))

        release_computational_graph(model, poisson_input_rate, current_inputs)
        spikes = None; loss = None; current_inputs = None

    avg_batch_loss = np.mean(np.asarray(batch_losses, dtype=np.float))

    logger.log('batch losses: {}'.format(batch_losses))
    logger.log('avg_batch_loss: {}'.format(avg_batch_loss), {'train_i': train_i})
    logger.log(parameters=[train_i, avg_abs_grads])
    gen_inputs = None

    return avg_batch_loss, np.mean(np.asarray(avg_abs_grads, dtype=np.float)), batch_losses[-1]
