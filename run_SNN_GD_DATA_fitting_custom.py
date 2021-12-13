import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import IO
import Log
import data_util
import experiments
import model_util
import plot
import spike_metrics
from Models.GLIF import GLIF
from Models.GLIF_no_cell_types import GLIF_no_cell_types
from Models.LIF import LIF
from Models.LIF_no_cell_types import LIF_no_cell_types
from experiments import release_computational_graph

# target_data_fname = 'GT_LIF_N_4_seed_3_duration_120000.mat'
# target_data_path = data_util.prefix + data_util.target_data_path

target_data_path = data_util.prefix + data_util.sleep_data_path
sleep_data_files = ['exp108.mat', 'exp109.mat', 'exp124.mat', 'exp126.mat', 'exp138.mat', 'exp146.mat', 'exp147.mat']
class_lookup = { 'LIF_no_cell_types': LIF_no_cell_types, 'GLIF_no_cell_types': GLIF_no_cell_types }


def main(argv):
    print('Argument List:', str(argv))
    start_seed = 13
    num_seeds = 10
    model_class = LIF_no_cell_types
    # model_class = GLIF_no_cell_types
    sleep_exp_num = None
    lfn = 'frd'
    # lfn = 'vrd'

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]
    for i, opt in enumerate(opts):
        if opt == '-h':
            print('run_SNN_GD_DATA_fitting_custom -mt <model-type> -lfn <loss-fn> -sexp <sleep-data-exp-num> -nos <number-of-seeds>')
            sys.exit()
        elif opt in ("-mt", "--model-type"):
            model_class = class_lookup[str(args[i])]
            if model_class is None:
                raise NotImplementedError("Model class str not supported: {}".format(args[i]))
        elif opt in ("-sexp", "--sleep-data-exp-num"):
            sleep_exp_num = int(args[i])
        elif opt in ("-nos", "--number-of-seeds"):
            num_seeds = int(args[i])
        elif opt in ("-ss", "--start-seed"):
            start_seed = int(args[i])
        elif opt in ("-lfn", "--loss-fn"):
            lfn = str(args[i])

    if sleep_exp_num is None:
        data_files = sleep_data_files
    else:
        data_files = [sleep_data_files[sleep_exp_num]]

    for data_file in data_files:
        node_indices, spike_times, spike_indices = data_util.load_sparse_data(target_data_path + data_file)

        for random_seed in range(start_seed, start_seed+num_seeds):
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

            # N = 4
            N = len(np.unique(node_indices))
            t = 1200
            learn_rate = 0.001
            num_train_iter = 100
            plot_every = round(num_train_iter/20)
            bin_size = 100
            tau_vr = 4.
            # optim_class = torch.optim.SGD
            optim_class = torch.optim.Adam
            config_str = '$\\alpha={}$, lfn: {}, bin_size: {}, optim: {}'.format(learn_rate, lfn, bin_size, optim_class.__name__)

            timestamp = IO.dt_descriptor()
            logger = Log.Logger('{}_GD_{}.txt'.format('DATA', timestamp))
            writer = SummaryWriter('runs/' + timestamp)
            # current_uuid = 'GT' + '/' + model_class.__name__ + '/' + timestamp
            current_uuid = 'sleep_data_no_types' + '/' + model_class.__name__ + '/' + timestamp

            A_coeffs = [torch.randn((4,))]
            phase_shifts = [torch.rand((4,))]
            input_types = [1, 1, 1, 1]

            next_step_i = 0
            next_step_i, target_spikes = data_util.get_spike_train_matrix(next_step_i, t, spike_times, spike_indices, node_indices)
            # target_spikes = target_spikes.clone().detach()
            target_parameters = False

            params_model = experiments.draw_from_uniform(model_class.parameter_init_intervals, N)
            neuron_types = N * [1]
            n_inhib = int(N/3)
            neuron_types[-n_inhib:] = n_inhib * [-1]
            snn = model_class(parameters=params_model, N=N)

            fig_W_init = plot.plot_heatmap((snn.w.clone().detach()), ['W_syn_col', 'W_row'],
                                           uuid=current_uuid, exp_type='GD_test', fname='plot_heatmap_W_initial.png')

            optim_params = list(snn.parameters())
            optimiser = optim_class(optim_params, lr=learn_rate)

            losses = []; prev_write_index = -1
            weights = []
            model_parameter_trajectories = {}
            cur_params = snn.state_dict()
            for p_i, key in enumerate(cur_params):
                model_parameter_trajectories[key] = [cur_params[key].clone().detach().numpy()]
            for i in range(num_train_iter+1):
                optimiser.zero_grad()

                white_noise = torch.rand((t, N))
                current_inputs = white_noise
                if (next_step_i+t)>spike_indices[-1]:
                    next_step_i = 0
                next_step_i, target_spikes = data_util.get_spike_train_matrix(next_step_i, t, spike_times, spike_indices, node_indices)
                vs, spikes = model_util.feed_inputs_sequentially_return_tuple(snn, current_inputs)

                if lfn == 'frd':
                    loss = spike_metrics.firing_rate_distance(model_spikes=spikes, target_spikes=target_spikes)
                elif lfn == 'vrd':
                    loss = spike_metrics.van_rossum_dist(spikes=spikes, target_spikes=target_spikes, tau=tau_vr)
                elif lfn == 'frdvrd':
                    frd_loss = spike_metrics.firing_rate_distance(model_spikes=spikes, target_spikes=target_spikes)
                    vrd_loss = spike_metrics.van_rossum_dist(spikes=spikes, target_spikes=target_spikes, tau=tau_vr)
                    loss = frd_loss + vrd_loss
                elif lfn == 'custom':
                    loss = spike_metrics.correlation_metric_distance(spikes, target_spikes, bin_size=100)
                else:
                    raise NotImplementedError()

                loss.backward(retain_graph=True)

                optimiser.step()

                loss_data = loss.clone().detach().data
                losses.append(loss_data)
                print('loss: {}'.format(loss_data))
                writer.add_scalar('training_loss', scalar_value=loss_data, global_step=i)

                if i == 0 or i % plot_every == 0 or i == num_train_iter:
                    fig_spikes = plot.plot_spike_trains_side_by_side(spikes, target_spikes, uuid=current_uuid,
                                                        exp_type='GD_test', title='Test {} spike trains'.format(snn.__class__.__name__),
                                                        legend=['Initial', 'Target'], fname='spike_trains_train_iter_{}.png'.format(i))
                    fig_vs = plot.plot_neuron(vs.detach().data, uuid=current_uuid, exp_type='GD_test', fname='membrane_pots_train_i_{}.png'.format(i))
                    fig_heatmap = plot.plot_heatmap(snn.w.clone().detach(), ['W_syn_col', 'W_row'], uuid=current_uuid,
                                           exp_type='GD_test', fname='plot_heatmap_W_train_i_{}.png'.format(i))

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

            plot.plot_loss(losses, uuid=current_uuid, exp_type='GD_test',
                           custom_title='Loss {}, $\\alpha$={}, {}, bin_size={}'.format(lfn, learn_rate, optimiser.__class__.__name__, bin_size),
                           fname='plot_loss_test'+IO.dt_descriptor())

            plot.plot_spike_trains_side_by_side(spikes, target_spikes, uuid=current_uuid,
                                                exp_type='GD_test', title='Test {} spike trains'.format(snn.__class__.__name__),
                                                legend=['Fitted', 'Target'])

            _ = plot.plot_heatmap((snn.w.clone().detach()), ['W_syn_col', 'W_row'], uuid=current_uuid,
                                  exp_type='GD_test', fname='plot_heatmap_W_after_training.png')

            hard_thresh_spikes_sum = torch.round(spikes).sum()
            print('spikes sum: {}'.format(hard_thresh_spikes_sum))
            soft_thresh_spikes_sum = (spikes > 0.333).sum()
            zero_thresh_spikes_sum = (spikes > 0).sum()
            print('thresholded spikes sum: {}'.format(torch.round(spikes).sum()))
            print('=========avg. hard rate: {}'.format(1000*hard_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))
            print('=========avg. soft rate: {}'.format(1000*soft_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))
            print('=========avg. zero thresh rate: {}'.format(1000*zero_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))

            # 🍝 weights across iterations plot.
            plot.plot_parameter_inference_trajectories_2d({'w': weights}, target_params=False,
            # plot.plot_parameter_inference_trajectories_2d({'w': weights}, target_params={'w': snn_target.w.detach().flatten().numpy() },
                                                          uuid=current_uuid,
                                                          exp_type='GD_test',
                                                          param_names=['w'],
                                                          custom_title='Test weights plot',
                                                          fname='test_weights_inference_trajectories')

            parameter_names = snn.free_parameters
            plot.plot_parameter_inference_trajectories_2d(model_parameter_trajectories,
                                                          uuid=current_uuid,
                                                          exp_type='GD_test',
                                                          target_params=target_parameters,
                                                          param_names=parameter_names,
                                                          custom_title='Inferred parameters across training iterations',
                                                          fname='inferred_param_trajectories_{}_exp_num_{}_train_iters_{}'
                                                          .format(snn.__class__.__name__, None, i))

            IO.save(snn, loss={'losses': losses}, uuid=current_uuid, fname='snn_model_target_GD_test')

            logger.log('snn.parameters(): {}'.format(snn.parameters()), list(snn.parameters()))
            logger.log('model_parameter_trajectories: ', model_parameter_trajectories)


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
