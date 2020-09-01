import torch
from torch.nn.functional import poisson_nll_loss

from experiments import poisson_input


def test_poisson_NLL():
    tar_spikes = poisson_input(1. * torch.ones((3,)), t=100, N=3)
    model_spikes = poisson_input(1. * torch.ones((3,)), t=100, N=3)
    print('num of sample model spikes: {}'.format(model_spikes.sum()))
    print('num of sample target spikes: {}'.format(tar_spikes.sum()))

    loss = poisson_nll_loss(model_spikes, tar_spikes)
    print('poisson nll.: {}'.format(loss))

    zeros = torch.zeros_like(tar_spikes)
    print('num of spikes in zeros: {}'.format(zeros.sum()))

    loss_zeros = poisson_nll_loss(zeros.clone(), zeros)
    print('poisson nll.: {}'.format(loss_zeros))
    assert loss_zeros == 1.0, "distance between silent trains should be approximately zero. was: {}".format(loss_zeros)

    loss_model_spikes_zeros = poisson_nll_loss(model_spikes, zeros)
    print('loss_model_spikes_zeros: {}'.format(loss_model_spikes_zeros))
    assert loss_model_spikes_zeros > loss_zeros, "spikes should result in greater loss with spikes than no spikes with no spikes as target"


# --------------------------------------
test_poisson_NLL()
