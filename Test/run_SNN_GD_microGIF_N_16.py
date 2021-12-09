import sys

import numpy as np
import torch
import torch.tensor as T
from torch.utils.tensorboard import SummaryWriter

import IO
import Log
import PDF_metrics
import experiments
import model_util
import plot
from Models.microGIF import microGIF
from TargetModels.TargetModelMicroGIF import micro_gif_populations_model_full_size
from experiments import release_computational_graph, draw_from_uniform

target_timestamp = '12-09_14-56-17-312'
for lfn_str in ['bernoulli_nll', 'poisson_nll']:
    for random_seed in range(3, 24):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        # pop_sizes, snn = TargetModelMicroGIF.micro_gif_populations_model_full_size(random_seed=random_seed)

        # pop_sizes, snn_target = micro_gif_populations_model_full_size(random_seed=random_seed)
        pop_sizes, _ = micro_gif_populations_model_full_size(random_seed=random_seed)
        fname = 'snn_model_target_GD_test'
        load_data = torch.load(IO.PATH + microGIF.__name__ + '/' + target_timestamp + '/' + fname + IO.fname_ext)
        snn_target = load_data['model']

        N = snn_target.N
        t = 1200
        timestamp = IO.dt_descriptor()
        # lfn_str = 'bernoulli_nll'
        lfn_str = 'poisson_nll'
        learn_rate = 0.01
        num_train_iter = 100
        plot_every = round(num_train_iter / 20)

        # neurons_coeff = torch.cat([T(pop_sizes[0] * [0.]), T(pop_sizes[1] * [0.]), T(pop_sizes[2] * [0.25]), T(pop_sizes[3] * [0.1])])
        neurons_coeff = torch.cat([T(8 * [0.25]), T(8 * [0.1])])

        A_coeff_1 = torch.randn((4,))
        A_coeff_2 = torch.randn((4,))
        phase_shifts_1 = torch.rand((4,))
        phase_shifts_2 = phase_shifts_1 + torch.rand((4,))

        inputs_1 = experiments.white_noise_sum_of_sinusoids(t=t, A_coeff=A_coeff_1, phase_shifts=phase_shifts_1)
        inputs_2 = experiments.white_noise_sum_of_sinusoids(t=t, A_coeff=A_coeff_2, phase_shifts=phase_shifts_2)

        current_inputs = torch.vstack([inputs_1, inputs_2])
        for _ in range(N - 2):
            current_inputs = torch.vstack([current_inputs, torch.rand((1, t)).clamp(0., 1.)])
        current_inputs = current_inputs.T
        current_inputs = torch.tensor(current_inputs.clone().detach(), requires_grad=True)
        target_s_lambdas, target_model_spiketrain, target_vs = model_util.feed_inputs_sequentially_return_args(snn_target, current_inputs)
        target_spikes = target_model_spiketrain.clone().detach()

        params_model = draw_from_uniform(microGIF.parameter_init_intervals, N)
        params_model['N'] = N

        snn = microGIF(N=N, parameters=params_model)
        optim_params = list(snn.parameters())
        # optimiser = torch.optim.SGD(optim_params, lr=learn_rate)
        optimiser = torch.optim.Adam(optim_params, lr=learn_rate)

        timestamp = IO.dt_descriptor()
        logger = Log.Logger('{}_GD_{}.txt'.format(snn.__class__.__name__, timestamp))
        writer = SummaryWriter('runs/' + timestamp)

        fig_inputs = plot.plot_neuron(current_inputs.detach().data, uuid=snn.__class__.__name__ + '/{}'.format(timestamp),
                                      exp_type='GD_test', fname='train_inputs.png')
        fig_tar_vs = plot.plot_neuron(target_vs.detach().data, uuid=snn.__class__.__name__ + '/{}'.format(timestamp),
                                      exp_type='GD_test',
                                      fname='membrane_pots_target.png')
        tar_W_heatmap_fig = plot.plot_heatmap(snn_target.w.detach().numpy(), ['W_syn_col', 'W_row'],
                                              uuid=snn.__class__.__name__ + '/{}'.format(timestamp),
                                              exp_type='GD_test', fname='plot_heatmap_W_target.png',
                                              v_min=-10, v_max=10)

        writer.add_figure('Training input', fig_inputs)
        writer.add_figure('Target W heatmap', tar_W_heatmap_fig)
        writer.add_figure('Target vs', fig_tar_vs)

        losses = []; prev_write_index = -1
        weights = []
        model_parameter_trajectories = {}
        cur_params = snn.state_dict()
        for p_i, key in enumerate(cur_params):
            model_parameter_trajectories[key] = [cur_params[key].clone().detach().numpy()]
        for i in range(num_train_iter+1):
            optimiser.zero_grad()

            # current_inputs = sine_modulated_white_noise(t=t, N=snn.N, neurons_coeff=neurons_coeff)
            # current_inputs = sine_input(t=t, N=snn.N, neurons_coeff=neurons_coeff)
            # current_inputs.retain_grad()

            inputs_1 = experiments.white_noise_sum_of_sinusoids(t=t, A_coeff=A_coeff_1, phase_shifts=phase_shifts_1)
            inputs_2 = experiments.white_noise_sum_of_sinusoids(t=t, A_coeff=A_coeff_2, phase_shifts=phase_shifts_2)

            current_inputs = torch.vstack([inputs_1, inputs_2])
            for _ in range(N - 2):
                current_inputs = torch.vstack([current_inputs, torch.rand((1, t)).clamp(0., 1.)])
            current_inputs = current_inputs.T
            current_inputs = torch.tensor(current_inputs.clone().detach(), requires_grad=True)

            s_lambdas, model_spiketrain, vs = model_util.feed_inputs_sequentially_return_args(snn, current_inputs)

            if lfn_str == 'bernoulli_nll':
                loss = PDF_metrics.bernoulli_nll(spike_probabilities=s_lambdas, target_spikes=target_spikes)
            elif lfn_str == 'poisson_nll':
                loss = PDF_metrics.poisson_nll(spike_probabilities=s_lambdas, target_spikes=target_spikes, bin_size=100)
            else:
                raise NotImplementedError()

            loss.backward(retain_graph=True)

            optimiser.step()

            loss_data = loss.clone().detach().data
            losses.append(loss_data)
            print('loss: {}'.format(loss_data))
            writer.add_scalar('training_loss', scalar_value=loss_data, global_step=i)

            if i == 0 or i % plot_every == 0 or i == num_train_iter:
                fig_spikes = plot.plot_spike_trains_side_by_side(model_spiketrain, target_spikes, uuid=snn.__class__.__name__ + '/{}'.format(timestamp),
                                                    exp_type='GD_test', title='Test {} spike trains'.format(snn.__class__.__name__),
                                                    legend=['Initial', 'Target'], fname='spike_trains_train_iter_{}.png'.format(i))
                fig_vs = plot.plot_neuron(vs.detach().data, uuid=snn.__class__.__name__ + '/{}'.format(timestamp), exp_type='GD_test', fname='membrane_pots_train_i_{}.png'.format(i))
                fig_heatmap = plot.plot_heatmap(snn.w.clone().detach(), ['W_syn_col', 'W_row'], uuid=snn.__class__.__name__ + '/{}'.format(timestamp),
                                       exp_type='GD_test', fname='plot_heatmap_W_train_i_{}.png'.format(i), v_min=-10, v_max=10)

                # writer.add_scalars('training_loss', { 'losses': torch.tensor(losses[prev_write_index:]) }, i)
                # for loss_i in range(len(losses) - prev_write_index):
                #     writer.add_scalar('training_loss', scalar_value=losses[prev_write_index+loss_i], global_step=prev_write_index+loss_i)
                # prev_write_index = i

                weights.append((snn.w.clone().detach()).flatten().numpy())
                # weights.append(np.mean(snn.w.clone().detach().numpy(), axis=1))

                for p_i, param in enumerate(list(snn.parameters())):
                    print('grad for param #{}: {}'.format(p_i, param.grad))

                # ...log a Matplotlib Figure showing the model's predictions on a random mini-batch
                writer.add_figure('Model spikes vs. target spikes', fig_spikes, global_step=i)
                writer.add_figure('Model membrane potentials', fig_vs, global_step=i)
                writer.add_figure('Weights heatmap', fig_heatmap, global_step=i)

            cur_params = snn.state_dict()
            for p_i, key in enumerate(cur_params):
                model_parameter_trajectories[key].append(cur_params[key].clone().detach().numpy())

            release_computational_graph(snn, current_inputs)
            loss = None; current_inputs = None


        plot.plot_loss(losses, uuid=snn.__class__.__name__ + '/{}'.format(timestamp), exp_type='GD_test',
                       custom_title='Loss {}, $\\alpha$={}, {}'.format(lfn_str, learn_rate, optimiser.__class__.__name__),
                       fname='plot_loss_test'+IO.dt_descriptor())

        plot.plot_spike_trains_side_by_side(model_spiketrain, target_spikes, uuid=snn.__class__.__name__ + '/{}'.format(timestamp), exp_type='GD_test',
                                            title='Test {} spike trains'.format(snn.__class__.__name__),
                                            legend=['Fitted', 'Target'])

        _ = plot.plot_heatmap((snn.w.clone().detach()), ['W_syn_col', 'W_row'], uuid=snn.__class__.__name__+'/{}'.format(timestamp),
                              exp_type='GD_test', fname='plot_heatmap_W_after_training.png', v_min=-10, v_max=10)

        # spikes = model_util.feed_inputs_sequentially_return_args(snn, inputs)
        # print('snn weights: {}'.format(snn.w))
        hard_thresh_spikes_sum = torch.round(model_spiketrain).sum()
        print('spikes sum: {}'.format(hard_thresh_spikes_sum))
        soft_thresh_spikes_sum = (model_spiketrain > 0.333).sum()
        zero_thresh_spikes_sum = (model_spiketrain > 0).sum()
        print('thresholded spikes sum: {}'.format(torch.round(model_spiketrain).sum()))
        print('=========avg. hard rate: {}'.format(1000*hard_thresh_spikes_sum / (model_spiketrain.shape[1] * model_spiketrain.shape[0])))
        print('=========avg. soft rate: {}'.format(1000*soft_thresh_spikes_sum / (model_spiketrain.shape[1] * model_spiketrain.shape[0])))
        print('=========avg. zero thresh rate: {}'.format(1000*zero_thresh_spikes_sum / (model_spiketrain.shape[1] * model_spiketrain.shape[0])))

        # 🍝 weights across iterations plot.
        # plot.plot_parameter_inference_trajectories_2d({'w': weights}, target_params={'w': snn_target.w.detach().flatten().numpy() },
        plot.plot_parameter_inference_trajectories_2d({'w': weights}, target_params={'w': snn_target.w.detach().flatten().numpy() },
                                                      uuid=snn.__class__.__name__ + '/' + timestamp,
                                                      exp_type='GD_test',
                                                      param_names=['w'],
                                                      custom_title='Test weights plot',
                                                      fname='test_weights_inference_trajectories')

        parameter_names = snn.free_parameters
        plot.plot_parameter_inference_trajectories_2d(model_parameter_trajectories,
                                                      uuid=snn.__class__.__name__ + '/' + timestamp,
                                                      exp_type='GD_test',
                                                      target_params=snn_target.state_dict(),
                                                      param_names=parameter_names,
                                                      custom_title='Inferred parameters across training iterations',
                                                      fname='inferred_param_trajectories_{}_exp_num_{}_train_iters_{}'
                                                      .format(snn.__class__.__name__, None, i))

        IO.save(snn, loss={'losses': losses}, uuid=snn.__class__.__name__ + '/' + timestamp, fname='snn_model_target_GD_test')

        logger.log('snn.parameters(): {}'.format(snn.parameters()), list(snn.parameters()))
        logger.log('model_parameter_trajectories: ', model_parameter_trajectories)

sys.exit(0)
