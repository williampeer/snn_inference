import sys

import numpy as np
import torch
import torch.tensor as T

import IO
import PDF_metrics
import model_util
import plot
from Models.microGIF_transposed import microGIF_transposed
from TargetModels.TargetModelMicroGIF import get_low_dim_micro_GIF_transposed
from experiments import sine_modulated_white_noise, sine_input, release_computational_graph, draw_from_uniform
from plot import plot_spike_train_projection

for random_seed in range(3, 4):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    # pop_sizes, snn = TargetModelMicroGIF.micro_gif_populations_model_full_size(random_seed=random_seed)
    pop_sizes, snn_target = get_low_dim_micro_GIF_transposed(random_seed=random_seed)

    N = snn_target.N
    t = 4800
    # neurons_coeff = torch.cat([T(pop_sizes[0] * [0.]), T(pop_sizes[1] * [0.]), T(pop_sizes[2] * [0.25]), T(pop_sizes[3] * [0.1])])
    neurons_coeff = torch.cat([T(2 * [0.25]), T(2 * [0.1])])
    # sample_inputs = sine_modulated_white_noise(t=t, N=snn.N, neurons_coeff=neurons_coeff)
    sample_inputs = sine_input(t=t, N=snn_target.N, neurons_coeff=neurons_coeff)
    print('- SNN test for class {} -'.format(snn_target.__class__.__name__))
    print('#inputs: {}'.format(sample_inputs.sum()))
    _, target_spikes = model_util.feed_inputs_sequentially_return_tuple(snn_target, sample_inputs)
    target_spikes = target_spikes.clone().detach()

    params_model = draw_from_uniform(microGIF_transposed.parameter_init_intervals, N)
    params_model['N'] = N
    params_model['R_m'] = snn_target.R_m.clone().detach()

    snn = microGIF_transposed(N=N, parameters=params_model, neuron_types=torch.tensor([1., 1., -1., -1.]))
    optim_params = list(snn.parameters())
    learn_rate = 0.03
    # optimiser = torch.optim.SGD(optim_params, lr=learn_rate)
    optimiser = torch.optim.Adam(optim_params, lr=learn_rate)

    losses = []
    for i in range(10):
        optimiser.zero_grad()

        # current_inputs = sine_modulated_white_noise(t=t, N=snn.N, neurons_coeff=neurons_coeff)
        current_inputs = sine_input(t=t, N=snn.N, neurons_coeff=neurons_coeff)
        current_inputs.retain_grad()

        spike_probs, spikes = model_util.feed_inputs_sequentially_return_tuple(snn, current_inputs)

        if i == 0:
            plot.plot_spike_trains_side_by_side(spikes, target_spikes, uuid='test', exp_type='GD_test',
                                                title='Test {} spike trains'.format(snn.__class__.__name__),
                                                legend=['Fitted', 'Target'], fname='spike_trains_before_training.png')

        # loss = spike_metrics.firing_rate_distance(spikes, sample_targets)
        # m = torch.distributions.bernoulli.Bernoulli(spike_probs)
        # loss = -m.log_prob(sample_targets).sum()
        # loss = PDF_metrics.bernoulli_nll(spike_probabilities=spike_probs, target_spikes=sample_targets)
        loss = PDF_metrics.poisson_nll(spike_probabilities=spike_probs, target_spikes=target_spikes, bin_size=100)

        loss.backward(retain_graph=True)
        # loss.backward()

        for p_i, param in enumerate(list(snn.parameters())):
            print('grad for param #{}: {}'.format(p_i, param.grad))

        optimiser.step()

        losses.append(loss.clone().detach().data)

        release_computational_graph(snn, current_inputs)
        loss = None; current_inputs = None


    plot.plot_loss(losses, uuid='test', exp_type='GD_test',
                   custom_title='Loss {}, $\\alpha$={}, {}'.format('poisson_nll', learn_rate, optimiser.__class__.__name__),
                   fname='plot_loss_test'+IO.dt_descriptor())

    plot.plot_spike_trains_side_by_side(spikes, target_spikes, uuid='test', exp_type='GD_test',
                                        title='Test {} spike trains'.format(snn.__class__.__name__),
                                        legend=['Fitted', 'Target'])

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
    # plot_spike_train_projection(spikes, fname='test_projection_{}_ext_input'.format(snn.__class__.__name__) + '_' + str(random_seed))

sys.exit(0)
