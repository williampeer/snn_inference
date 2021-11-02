import sys

import numpy as np
import torch
import torch.tensor as T

import PDF_metrics
import model_util
from TargetModels import TargetModelMicroGIF
from experiments import sine_modulated_white_noise
from plot import plot_spike_train_projection

# num_pops = 2
num_pops = 4
pop_size = 4
# pop_size = 2
# pop_size = 1

for random_seed in range(3, 4):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    pop_sizes, snn = TargetModelMicroGIF.micro_gif_populations_model_full_size(random_seed=random_seed)

    N = snn.N
    t = 4800
    neurons_coeff = torch.cat([T(pop_sizes[0] * [0.]), T(pop_sizes[1] * [0.]), T(pop_sizes[2] * [0.25]), T(pop_sizes[3] * [0.1])])
    sample_inputs = sine_modulated_white_noise(t=t, N=snn.N, neurons_coeff=neurons_coeff)
    print('- SNN test for class {} -'.format(snn.__class__.__name__))
    print('#inputs: {}'.format(sample_inputs.sum()))
    _, sample_targets = model_util.feed_inputs_sequentially_return_tuple(snn, sample_inputs)
    sample_targets = sample_targets.clone().detach()

    optim_params = list(snn.parameters())
    optimiser = torch.optim.SGD(optim_params, lr=0.015)
    optimiser.zero_grad()

    for i in range(3):
        current_inputs = sine_modulated_white_noise(t=t, N=snn.N, neurons_coeff=neurons_coeff)
        current_inputs.retain_grad()

        spike_probs, spikes = model_util.feed_inputs_sequentially_return_tuple(snn, current_inputs)

        # loss = spike_metrics.firing_rate_distance(spikes, sample_targets)
        # m = torch.distributions.bernoulli.Bernoulli(spike_probs)
        # loss = -m.log_prob(sample_targets).sum()
        # loss = PDF_metrics.bernoulli_nll(spike_probabilities=spike_probs, target_spikes=sample_targets)
        loss = PDF_metrics.poisson_nll(spike_probabilities=spike_probs, target_spikes=sample_targets, bin_size=100)

        loss.backward(retain_graph=True)
        # loss.backward()

        for p_i, param in enumerate(list(snn.parameters())):
            print('grad for param #{}: {}'.format(p_i, param.grad))

    optimiser.step()
    # release_computational_graph(snn, False, current_inputs)
    # spikes = None; loss = None; current_inputs = None

    # spikes = model_util.feed_inputs_sequentially_return_spike_train(snn, inputs)
    # print('snn weights: {}'.format(snn.w))
    hard_thresh_spikes_sum = torch.round(spikes).sum()
    print('spikes sum: {}'.format(hard_thresh_spikes_sum))
    soft_thresh_spikes_sum = (spikes > 0.333).sum()
    zero_thresh_spikes_sum = (spikes > 0).sum()
    print('thresholded spikes sum: {}'.format(torch.round(spikes).sum()))
    print('=========avg. hard rate: {}'.format(1000*hard_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))
    print('=========avg. soft rate: {}'.format(1000*soft_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))
    print('=========avg. zero thresh rate: {}'.format(1000*zero_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))
    plot_spike_train_projection(spikes, fname='test_projection_{}_ext_input'.format(snn.__class__.__name__) + '_' + str(random_seed))

sys.exit(0)
