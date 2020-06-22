import matplotlib.pyplot as plt
import torch
import numpy as np

from scipy.stats import gaussian_kde

import IO


def plot_spiketrain(spike_history, title, uuid, exp_type='default', fname='spiketrain_test'):
    data = {'spike_history': spike_history, 'title': title, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='plot_spiketrain')

    plt.figure()
    # assuming neuron indices to be columns and reshaping to rows for plotting
    time_indices = torch.reshape(torch.arange(spike_history.shape[0]), (spike_history.shape[0], 1))
    # ensure binary values:
    spike_history = torch.round(spike_history)
    neuron_spike_times = spike_history * time_indices

    for neuron_i in range(spike_history.shape[1]):
        if neuron_spike_times[:, neuron_i].nonzero().sum() > 0:
            plt.plot(torch.reshape(neuron_spike_times[:, neuron_i].nonzero(), (1, -1)),
                     neuron_i+1, '.k', markersize=4.0)

    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    plt.yticks(range(neuron_i+2))
    plt.title(title)

    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists(full_path)
    plt.savefig(fname=full_path + fname)
    # plt.show()
    plt.close()


def plot_spiketrains_side_by_side(model_spikes, target_spikes, uuid, exp_type='default', title=False, fname=False):
    assert model_spikes.shape[0] > model_spikes.shape[1], \
        "assert one node per column, one bin per row. spikes shape: {}".format(model_spikes.shape)
    assert model_spikes.shape[0] == target_spikes.shape[0], \
        "assert same number of bins / time interval. m_spikes.shape: {}, target shape: {}".format(model_spikes.shape, target_spikes.shape)

    if not fname:
        fname = 'spiketrains_' + IO.dt_descriptor()

    data = {'model_spikes': model_spikes, 'target_spikes': target_spikes, 'exp_type': exp_type, 'title': title, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='plot_spiketrains_side_by_side')

    fig = plt.figure()
    time_indices = torch.reshape(torch.arange(model_spikes.shape[0]), (model_spikes.shape[0], 1)).float()

    # ensure binary values:
    model_spike_history = torch.round(model_spikes)
    target_spike_history = torch.round(target_spikes)
    model_spike_times = model_spike_history * time_indices
    target_spike_times = target_spike_history * time_indices

    plt.plot(0, -1, '.b')
    plt.plot(0, -1, '.g')
    plt.legend(['Model', 'Target'])

    for neuron_i in range(model_spike_history.shape[1]):
        if model_spike_times[:, neuron_i].nonzero().sum() > 0:
            plt.plot(torch.reshape(model_spike_times[:, neuron_i].nonzero(), (1, -1)).numpy(),
                     neuron_i + 1.1, '.b', markersize=4.0, label='Model')
        if target_spike_times[:, neuron_i].nonzero().sum() > 0:
            plt.plot(torch.reshape(target_spike_times[:, neuron_i].nonzero(), (1, -1)).numpy(),
                     neuron_i + 0.9, '.g', markersize=4.0, label='Target')

    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    plt.yticks(range(1, neuron_i + 2))
    plt.ylim(0, neuron_i+2)
    if not title:
        title = 'Spiketrains side by side'
    plt.title(title)

    full_path = './figures/' + exp_type + '/' +  uuid + '/'
    IO.makedir_if_not_exists(full_path)
    fig.savefig(fname=full_path + fname)
    # plt.show()
    plt.close()


def plot_neuron(membrane_potentials_through_time, title='Neuron activity', fname_ext=False):
    data = {'membrane_potentials_through_time': membrane_potentials_through_time, 'title': title}
    IO.save_plot_data(data=data, uuid='test_uuid', plot_fn='plot_neuron')

    plt.figure()
    plt.plot(torch.arange(membrane_potentials_through_time.shape[0]), membrane_potentials_through_time)
    plt.title(title)
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential')
    # plt.show()
    fname = './figures/test_neuron'
    if fname_ext:
        fname = fname + fname_ext
    plt.savefig(fname=fname)
    plt.close()


def plot_losses(training_loss, test_loss, uuid, exp_type='default', custom_title=False, fname=False):
    if not fname:
        fname = 'training_and_test_loss'+IO.dt_descriptor()
    data = {'training_loss': training_loss, 'test_loss': test_loss, 'exp_type': exp_type, 'custom_title': custom_title, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='plot_losses')

    plt.figure()
    plt.plot(training_loss)
    plt.plot(test_loss)
    plt.legend(['Training loss', 'Test loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.xticks(range(len(loss_arr+1)))
    if custom_title:
        plt.title(custom_title)
    else:
        plt.title('Training and test set loss')

    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists(full_path)
    plt.savefig(fname=full_path + fname)
    # plt.show()
    plt.close()


def plot_losses_nodes(batch_loss_per_node, uuid, exp_type='default', custom_title=False, fname=False):
    if not fname:
        fname = 'batch_loss_per_node'+IO.dt_descriptor()
    data = {'batch_loss_per_node': batch_loss_per_node, 'exp_type': exp_type, 'custom_title': custom_title, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='plot_losses_nodes')

    plt.figure()
    for b_loss in batch_loss_per_node:
        plt.plot(b_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.xticks(range(len(loss_arr+1)))
    if custom_title:
        plt.title(custom_title)
    else:
        plt.title('Batch loss')

    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists(full_path)
    plt.savefig(fname=full_path + fname)
    # plt.show()
    plt.close()


def calculate_kde(p1, p2, logger):
    data = np.vstack([p1, p2])

    std_0 = np.std(p1)
    std_1 = np.std(p2)
    if std_0 == 0 or std_1 == 0:
        raise ArithmeticError("Std was zero - plot point along axis.")

    kde = gaussian_kde(data)

    # evaluate on a regular grid
    std_coeff = 4.0
    x_min = np.mean(p1) - std_coeff * std_0; x_max = np.mean(p1) + std_coeff * std_0
    y_min = np.mean(p2) - std_coeff * std_1; y_max = np.mean(p2) + std_coeff * std_1

    xgrid = np.linspace(x_min, x_max, 40)
    ygrid = np.linspace(y_min, y_max, 40)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)

    Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

    return Z, Xgrid, x_min, x_max, y_min, y_max


def plot_parameter_pair_with_variance(p1_means, p2_means, target_params, path, custom_title=False, logger=False):
    try:
        Z, Xgrid, x_min, x_max, y_min, y_max = calculate_kde(p1_means, p2_means, logger)

        plt.figure()
        # Plot the result as an image
        plt.imshow(Z.reshape(Xgrid.shape),
                   origin='lower', aspect='auto',
                   extent=[x_min, x_max, y_min, y_max],
                   cmap='Blues')
        cb = plt.colorbar()
        cb.set_label("density")

        if target_params:
            plt.plot(target_params[0], target_params[1], 'oy')
            plt.legend(['True value'])

        plt.xlabel('Parameter 1')
        plt.ylabel('Parameter 2')
        if custom_title:
            plt.title(custom_title)
        else:
            plt.title('Inferred parameter distributions')

        # plt.imsave(fname, image)
        plt.savefig(fname=path)
        # plt.show()
        plt.close()
    except:
        if not logger:
            print('WARN: Error calculating the kde. params: {}. {}'.format(p1_means, p2_means))
        else:
            logger.log(['plot.plot_parameter_pair_with_variance'], 'WARN: Error calculating the kde. params: {}. {}'.format(p1_means, p2_means))


def decompose_param_plot(param_2D, target_params, path):
    params_by_exp = np.array(param_2D).T
    num_of_parameters = params_by_exp.shape[0]
    print('in decompose_param_plot.. params_by_exp: {}'.format(params_by_exp))

    fig, axs = plt.subplots(nrows=num_of_parameters-1, ncols=num_of_parameters-1)
    [axi.set_axis_off() for axi in axs.ravel()]

    for i in range(num_of_parameters):
        for j in range(i + 1, num_of_parameters):
            # 2D plot KDE between p_i and p_j
            cur_ax = axs[i,j-1]
            try:
                Z, Xgrid, x_min, x_max, y_min, y_max = calculate_kde(params_by_exp[i], params_by_exp[j], False)

                cur_ax.imshow(Z.reshape(Xgrid.shape),
                                  origin='lower', aspect='auto',
                                  extent=[x_min, x_max, y_min, y_max],
                                  cmap='Blues')
                if target_params and len(target_params) > np.max([i, j]):
                    cur_ax.plot(target_params[0][i], target_params[0][j], 'oy', markersize=2.0)
            except ArithmeticError:
                cur_ax.plot(params_by_exp[i], params_by_exp[j], 'xb', markersize=3.5)
                if target_params and len(target_params) > np.max([i, j]):
                    cur_ax.plot(target_params[0][i], target_params[0][j], 'oy', markersize=2.0)
            except:
                print('WARN: Failed to calculate KDE for param.s: {}, {}'.format(params_by_exp[i], params_by_exp[j]))

    fig.suptitle('Decomposed KDE pairs for N-dimensional parameter')
    if not path:
        path = './figures/{}/{}/param_subplot_inferred_params_{}'.format('default', 'test_uuid', IO.dt_descriptor())
    fig.savefig(path)
    plt.close()


def plot_all_param_pairs_with_variance(param_means, target_params, exp_type, uuid, fname, custom_title, logger):
    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists(full_path)

    if not fname:
        fname = 'new_inferred_params_{}'.format(IO.dt_descriptor())
    path = full_path + fname

    number_of_parameters = len(param_means.values())
    for plot_i in range(number_of_parameters):  # assuming a dict., for all parameter combinations
        for plot_j in range(plot_i + 1, number_of_parameters):
            cur_tar_params = False
            if target_params and len(target_params) > np.max([plot_i, plot_j]):
                cur_tar_params = [target_params[plot_i], target_params[plot_j]]

            cur_p_i = np.array(param_means[plot_i])
            cur_p_j = np.array(param_means[plot_j])
            # silently fail for 3D params (weights)
            if len(cur_p_i.shape) == 2:
                cur_tar = False
                if target_params and len(target_params) > plot_i:
                    cur_tar = target_params[plot_i]
                decompose_param_plot(cur_p_i, cur_tar, path+'_param_{}'.format(plot_i))
            if len(cur_p_j.shape) == 2:
                cur_tar = False
                if target_params and len(target_params) > plot_j:
                    cur_tar = target_params[plot_j]
                decompose_param_plot(cur_p_j, cur_tar, path+'_param_{}'.format(plot_j))
            if len(cur_p_i.shape) == 1 and len(cur_p_j.shape) == 1:
                plot_parameter_pair_with_variance(cur_p_i, cur_p_j, target_params=cur_tar_params,
                                                  path=path+'_i_j_{}_{}'.format(plot_i, plot_j),
                                                  custom_title=custom_title, logger=logger)
