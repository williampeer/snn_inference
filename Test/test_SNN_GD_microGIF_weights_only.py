import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import IO
import PDF_metrics
import experiments
import model_util
import plot
from Models.microGIF_weights_only import microGIF_weights_only
from TargetModels.TargetModelMicroGIF import get_low_dim_micro_GIF_transposed
from experiments import release_computational_graph

start_seed = 6
num_seeds = 2
for random_seed in range(start_seed, start_seed+num_seeds):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    # pop_sizes, snn = TargetModelMicroGIF.micro_gif_populations_model_full_size(random_seed=random_seed)
    pop_sizes, snn_target = get_low_dim_micro_GIF_transposed(random_seed=random_seed)

    N = snn_target.N
    t = 1200
    learn_rate = 0.05
    num_train_iter = 1000
    plot_every = 50
    bin_size = 100
    # optim_class = torch.optim.SGD(optfig_params, lr=learn_rate)
    optim_class = torch.optim.Adam
    # lfn = PDF_metrics.PDF_LFN.BERNOULLI
    lfn = PDF_metrics.PDF_LFN.POISSON
    config_str = '$\\alpha={}$, lfn: {}, bin_size: {}, optim: {}'.format(learn_rate, lfn.name, bin_size, optim_class.__name__)

    timestamp = IO.dt_descriptor()
    writer = SummaryWriter('runs/' + timestamp)

    # neurons_coeff = torch.cat([T(pop_sizes[0] * [0.]), T(pop_sizes[1] * [0.]), T(pop_sizes[2] * [0.25]), T(pop_sizes[3] * [0.1])])
    # neurons_coeff = torch.cat([T(2 * [0.25]), T(2 * [0.1])])

    A_coeff_1 = torch.randn((4,))
    A_coeff_2 = torch.randn((4,))
    phase_shifts_1 = torch.rand((4,))
    phase_shifts_2 = phase_shifts_1 + torch.rand((4,))

    inputs_1 = experiments.white_noise_sum_of_sinusoids(t=t, A_coeff=A_coeff_1, phase_shifts=phase_shifts_1)
    # inputs_2 = experiments.white_noise_sum_of_sinusoids(t=t, A_coeff=A_coeff_2, phase_shifts=phase_shifts_2)

    current_inputs = torch.vstack([inputs_1, torch.zeros_like(inputs_1)])
    for _ in range(N - 2):
        current_inputs = torch.vstack([current_inputs, torch.zeros_like(inputs_1)])
    # current_inputs = torch.vstack([inputs_1, inputs_2])
    # for _ in range(N - 2):
    #     current_inputs = torch.vstack([current_inputs, torch.rand((1, t)).clamp(0., 1.)])
    current_inputs = current_inputs.T
    current_inputs = torch.tensor(current_inputs.clone().detach(), requires_grad=True)
    # target_spiketrain = experiments.auto_encode_input(current_inputs)

    # sample_inputs = sine_modulated_white_noise(t=t, N=snn.N, neurons_coeff=neurons_coeff)
    # sample_inputs = sine_input(t=t, N=snn_target.N, neurons_coeff=neurons_coeff)
    # print('- SNN test for class {} -'.format(snn_target.__class__.__name__))
    # print('#inputs: {}'.format(sample_inputs.sum()))
    _, target_spikes, target_vs = model_util.feed_inputs_sequentially_return_args(snn_target, current_inputs)
    target_spikes = target_spikes.clone().detach()

    # params_model = draw_from_uniform(microGIF.parameter_init_intervals, N)
    # params_model['N'] = N
    # params_model['R_m'] = snn_target.R_m.clone().detach()
    params_model = snn_target.get_parameters()

    snn = microGIF_weights_only(N=N, parameters=params_model, neuron_types=torch.tensor([1., 1., -1., -1.]))
    fig_W_init = plot.plot_heatmap(snn.w.detach().numpy() / 10., ['W_syn_col', 'W_row'], uuid=snn.__class__.__name__ + '/{}'.format(timestamp),
                                   exp_type='GD_test', fname='plot_heatmap_W_initial.png')

    optim_params = list(snn.parameters())
    optimiser = optim_class(optim_params, lr=learn_rate)

    fig_inputs = plot.plot_neuron(current_inputs.detach().data, uuid=snn.__class__.__name__ + '/{}'.format(timestamp),
                                  exp_type='GD_test', fname='train_inputs.png')
    fig_tar_vs = plot.plot_neuron(target_vs.detach().data, uuid=snn.__class__.__name__ + '/{}'.format(timestamp),
                                  exp_type='GD_test',
                                  fname='membrane_pots_target.png')
    tar_W_heatmap_fig = plot.plot_heatmap(snn_target.w.detach().numpy() / 10., ['W_syn_col', 'W_row'],
                                          uuid=snn.__class__.__name__ + '/{}'.format(timestamp),
                                          exp_type='GD_test', fname='plot_heatmap_W_target.png')

    writer.add_figure('Training input', fig_inputs)
    writer.add_figure('Target W heatmap', tar_W_heatmap_fig)
    writer.add_figure('Target vs', fig_tar_vs)
    # writer.add_figure('Initial model W heatmap', fig_W_init)

    losses = []; prev_write_index = -1
    weights = []
    for i in range(num_train_iter+1):
        optimiser.zero_grad()

        inputs_1 = experiments.white_noise_sum_of_sinusoids(t=t, A_coeff=A_coeff_1, phase_shifts=phase_shifts_1)
        # inputs_2 = experiments.white_noise_sum_of_sinusoids(t=t, A_coeff=A_coeff_2, phase_shifts=phase_shifts_2)

        current_inputs = torch.vstack([inputs_1, torch.zeros_like(inputs_1)])
        for _ in range(N - 2):
            current_inputs = torch.vstack([current_inputs, torch.zeros_like(inputs_1)])
        current_inputs = current_inputs.T
        current_inputs = torch.tensor(current_inputs.clone().detach(), requires_grad=True)

        spike_probs, spikes, vs = model_util.feed_inputs_sequentially_return_args(snn, current_inputs)

        # _, target_spikes, target_vs = model_util.feed_inputs_sequentially_return_args(snn_target, current_inputs)
        # target_spikes = target_spikes.clone().detach()

        if lfn == PDF_metrics.PDF_LFN.POISSON:
            loss = PDF_metrics.poisson_nll(spike_probabilities=spike_probs, target_spikes=target_spikes, bin_size=bin_size)
        elif lfn == PDF_metrics.PDF_LFN.BERNOULLI:
            loss = PDF_metrics.bernoulli_nll(spike_probabilities=spike_probs, target_spikes=target_spikes)
        else:
            raise NotImplementedError()

        loss.backward(retain_graph=True)

        weights.append(snn.w.clone().detach().flatten().numpy())
        # weights.append(np.mean(snn.w.clone().detach().numpy(), axis=1))

        # loss.backward()

        for p_i, param in enumerate(list(snn.parameters())):
            print('grad for param #{}: {}'.format(p_i, param.grad))

        optimiser.step()

        losses.append(loss.clone().detach().data)

        if i == 0 or i % plot_every == 0 or i == num_train_iter:
            fig_spikes = plot.plot_spike_trains_side_by_side(spikes, target_spikes, uuid=snn.__class__.__name__ + '/{}'.format(timestamp),
                                                exp_type='GD_test', title='Test {} spike trains'.format(snn.__class__.__name__),
                                                legend=['Initial', 'Target'], fname='spike_trains_train_iter_{}.png'.format(i))
            fig_vs = plot.plot_neuron(vs.detach().data, uuid=snn.__class__.__name__ + '/{}'.format(timestamp), exp_type='GD_test', fname='membrane_pots_train_i_{}.png'.format(i))
            fig_heatmap = plot.plot_heatmap(snn.w.detach().numpy() / 10., ['W_syn_col', 'W_row'], uuid=snn.__class__.__name__ + '/{}'.format(timestamp),
                                   exp_type='GD_test', fname='plot_heatmap_W_train_i_{}.png'.format(i))
            # plot.plot_neuron(target_vs.detach().data, uuid=snn.__class__.__name__+'/{}'.format(timestamp),
            # exp_type='GD_test', fname='membrane_pots_target_train_iter_{}.png'.format(i))

            # writer.add_scalars('training_loss', { 'losses': torch.tensor(losses[prev_write_index:]) }, i)
            for loss_i in range(len(losses) - prev_write_index):
                writer.add_scalar('training_loss', scalar_value=losses[prev_write_index+loss_i], global_step=prev_write_index+loss_i)
            prev_write_index = i

            # ...log a Matplotlib Figure showing the model's predictions on a random mini-batch
            writer.add_figure('Model spikes vs. target spikes', fig_spikes, global_step=i)
            writer.add_figure('Model membrane potentials', fig_vs, global_step=i)
            writer.add_figure('Weights heatmap', fig_heatmap, global_step=i)

        release_computational_graph(snn, current_inputs)
        loss = None; current_inputs = None


    plot.plot_loss(losses, uuid=snn.__class__.__name__+'/{}'.format(timestamp), exp_type='GD_test',
                   custom_title='Loss {}, $\\alpha$={}, {}, bin_size={}'.format(lfn.name, learn_rate, optimiser.__class__.__name__, bin_size),
                   fname='plot_loss_test'+IO.dt_descriptor())

    plot.plot_spike_trains_side_by_side(spikes, target_spikes, uuid=snn.__class__.__name__+'/{}'.format(timestamp),
                                        exp_type='GD_test', title='Test {} spike trains'.format(snn.__class__.__name__),
                                        legend=['Fitted', 'Target'])

    _ = plot.plot_heatmap(snn.w.detach().numpy() / 10., ['W_syn_col', 'W_row'], uuid=snn.__class__.__name__+'/{}'.format(timestamp),
                          exp_type='GD_test', fname='plot_heatmap_W_after_training.png')

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

    # 🍝 weights across iterations plot.
    plot.plot_parameter_inference_trajectories_2d({'w': weights}, target_params={'w': snn_target.w.detach().flatten().numpy() },
                                                  uuid=snn.__class__.__name__ + '/' + timestamp,
                                                  exp_type='GD_test',
                                                  param_names=['w'],
                                                  custom_title='Test weights plot',
                                                  fname='test_weights_inference_trajectories')

sys.exit(0)
