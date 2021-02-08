import matplotlib.pyplot as plt
import numpy as np
import torch

import IO

plt.rcParams.update({'font.size': 14})


def plot_spike_train(spike_train, title, uuid, exp_type='default', fname='spiketrain_test'):
    data = {'spike_history': spike_train, 'title': title, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='plot_spiketrain')

    plt.figure()
    # assuming neuron indices to be columns and reshaping to rows for plotting
    time_indices = torch.reshape(torch.arange(spike_train.shape[0]), (spike_train.shape[0], 1))
    # ensure binary values:
    spike_train = torch.round(spike_train)
    neuron_spike_times = spike_train * time_indices

    for neuron_i in range(spike_train.shape[1]):
        if neuron_spike_times[:, neuron_i].nonzero().sum() > 0:
            plt.plot(torch.reshape(neuron_spike_times[:, neuron_i].nonzero(), (1, -1)),
                     neuron_i+1, '.k', markersize=4.0)

    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    plt.yticks(range(neuron_i+2))
    plt.title(title)

    full_path = './figures/' + exp_type + '/' + uuid + '/'
    plt.savefig(fname=full_path + fname)
    # plt.show()
    plt.close()


def plot_spiketrains_side_by_side(model_spikes, target_spikes, uuid, exp_type='default', title='Spiketrains side by side',
                                  fname=False, legend=None, export=False):
    assert model_spikes.shape[0] > model_spikes.shape[1], \
        "assert one node per column, one bin per row. spikes shape: {}".format(model_spikes.shape)
    assert model_spikes.shape[0] == target_spikes.shape[0], \
        "assert same number of bins / time interval. m_spikes.shape: {}, target shape: {}".format(model_spikes.shape, target_spikes.shape)

    if not fname:
        fname = 'spiketrains_' + IO.dt_descriptor()

    if not export:
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
    if legend is not None:
        plt.legend(legend)
    else:
        plt.legend(['Model', 'Target'])

    for neuron_i in range(model_spike_history.shape[1]):
        if model_spike_times[:, neuron_i].nonzero().sum() > 0:
            plt.plot(torch.reshape(model_spike_times[:, neuron_i].nonzero(), (1, -1)).numpy(),
                     neuron_i + 1.1, '.b', markersize=4.0, label='Model')
    for neuron_i in range(target_spike_times.shape[1]):
        if target_spike_times[:, neuron_i].nonzero().sum() > 0:
            plt.plot(torch.reshape(target_spike_times[:, neuron_i].nonzero(), (1, -1)).numpy(),
                     neuron_i + 0.9, '.g', markersize=4.0-0.04*int(neuron_i/10), label='Target')

    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    if neuron_i > 20:
        plt.yticks(range(int((neuron_i+1)/10), neuron_i + 1, int((neuron_i+1)/10)))
    else:
        plt.yticks(range(1, neuron_i + 2))
    plt.ylim(0, neuron_i+2)
    plt.title(title)

    full_path = './figures/' + exp_type + '/' +  uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)
    fig.savefig(fname=full_path + fname)
    # plt.show()
    plt.close()


def plot_all_spiketrains(spikes_arr, uuid, exp_type='default', title=False, fname=False, legend=None):
    assert spikes_arr[0].shape[0] > spikes_arr[0].shape[1], \
        "assert one node per column, one bin per row. spikes shape: {}".format(spikes_arr[0].shape)

    if not fname:
        fname = 'spiketrains_' + IO.dt_descriptor()

    data = {'spikes_arr': spikes_arr, 'exp_type': exp_type, 'title': title, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='plot_spiketrains_side_by_side')

    fig = plt.figure()
    time_indices = torch.reshape(torch.arange(spikes_arr[0].shape[0]), (spikes_arr[0].shape[0], 1)).float()

    colours = ['.b', '.g', '.c', '.m', '.r']
    for i in range(len(spikes_arr)):
        plt.plot(0, -1, colours[i%len(colours)])

    for s_i in range(len(spikes_arr)):
        # ensure binary values:
        model_spike_history = torch.round(spikes_arr[s_i])
        model_spike_times = model_spike_history * time_indices

        for neuron_i in range(model_spike_history.shape[1]):
            if model_spike_times[:, neuron_i].nonzero().sum() > 0:
                plt.plot(torch.reshape(model_spike_times[:, neuron_i].nonzero(), (1, -1)).numpy(),
                         neuron_i + (1.0+0.5*0.15*len(spikes_arr)-0.15*s_i), colours[s_i%len(colours)], markersize=4.0)

    if legend is not None:
        plt.legend(legend, shadow=False, framealpha=0.5)

    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    plt.yticks(range(1, neuron_i + 2))
    plt.ylim(0, neuron_i+2)
    if not title:
        title = 'Spiketrains side by side'
    plt.title(title)

    full_path = './figures/' + exp_type + '/' +  uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)
    fig.savefig(fname=full_path + fname)
    # plt.show()
    plt.close()


def plot_neuron(membrane_potentials_through_time, uuid, exp_type='default', title='Neuron activity',
                ylabel='Membrane potential', fname='plot_neuron_test'):
    data = {'membrane_potentials_through_time': membrane_potentials_through_time, 'title': title, 'uuid': uuid,
            'exp_type': exp_type, 'ylabel': ylabel, 'fname': fname}
    IO.save_plot_data(data=data, uuid='test_uuid', plot_fn='plot_neuron')
    legend = []
    for i in range(len(membrane_potentials_through_time)):
        legend.append('N.{}'.format(i+1))
    plt.figure()
    plt.plot(torch.arange(membrane_potentials_through_time.shape[0]), membrane_potentials_through_time)
    plt.legend(legend, loc='upper left', ncol=4)
    plt.title(title)
    plt.xlabel('Time (ms)')
    plt.ylabel(ylabel)
    # plt.show()
    full_path = './figures/' + exp_type + '/' + uuid + '/'
    plt.savefig(fname=full_path + fname)


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
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)
    plt.savefig(fname=full_path + fname)
    # plt.show()
    plt.close()


def plot_avg_losses(avg_train_loss, train_loss_std, avg_test_loss, test_loss_std, uuid, exp_type='default', custom_title=False, fname=False):
    if not fname:
        fname = 'training_and_test_loss'+IO.dt_descriptor()
    data = {'avg_training_loss': avg_train_loss, 'avg_test_loss': avg_test_loss, 'exp_type': exp_type, 'custom_title': custom_title, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='plot_losses')

    plt.figure()
    xs_n = len(avg_train_loss)
    xs_n = 20
    plt.errorbar(np.linspace(1, xs_n, len(avg_train_loss)), y=avg_train_loss, yerr=train_loss_std)
    plt.errorbar(np.linspace(1, xs_n, len(avg_test_loss)), y=avg_test_loss, yerr=test_loss_std)
    plt.xticks(np.arange(11) * 2)

    plt.legend(['Training loss', 'Test loss'])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if custom_title:
        plt.title(custom_title)
    else:
        plt.title('Average training and test loss')

    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
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
    if custom_title:
        plt.title(custom_title)
    else:
        plt.title('Batch loss')

    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)
    plt.savefig(fname=full_path + fname)
    plt.show()
    plt.close()


def bar_plot_all_neuron_rates(rates, stds, bin_size, exp_type, uuid, fname, legends):
    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)

    data = {'rates': rates, 'stds': stds, 'exp_type': exp_type, 'uuid': uuid, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='bar_plot_neuron_rates')

    xs = np.linspace(1, rates[0].shape[0], rates[0].shape[0])
    width = 0.8/len(rates)
    max_rates = []; max_stds = []
    for i in range(len(rates)):
        print('plotting i: {}'.format(i))
        r = rates[i]; std = stds[i]
        plt.bar(xs-width+i*width, r.numpy(), yerr=std.numpy(), width=width)
        max_rates.append([np.max(r)])
        max_stds.append([np.max(std)])

    plt.legend(legends)
    r_max = np.max(max_rates); rstd_max = np.max(max_stds)
    summed_max = r_max + rstd_max

    plt.ylim(0, summed_max + rstd_max*0.05)
    plt.xticks(xs)

    plt.xlabel('Neuron')
    plt.ylabel('$Hz$')
    plt.title('Mean firing rate per neuron (bin size: {} ms)'.format(bin_size))

    # plt.show()
    plt.savefig(fname=full_path + fname)
    plt.close()


def heatmap_spike_train_correlations(corrs, axes, exp_type, uuid, fname, bin_size, custom_title=False, custom_label=False):
    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)

    data = {'corrs': corrs, 'exp_type': exp_type, 'uuid': uuid, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='heat_plot_spike_train_correlations')

    for row_i in range(corrs.shape[0]):
        for col_i in range(corrs.shape[1]):
            if np.isnan(corrs[row_i][col_i]):
                corrs[row_i][col_i] = 0.

    a = plt.imshow(corrs, cmap="PuOr", vmin=-1, vmax=1)
    cbar = plt.colorbar(a)
    if custom_label is not False:
        cbar.set_label(custom_label)
    else:
        cbar.set_label("correlation coeff.")
    # if custom_title is not False:
    #     plt.title(custom_title)
    # else:
    #     plt.title('Pairwise spike correlations (interval: {} ms)'.format(bin_size))
    plt.xticks(np.arange(0, len(corrs)))
    plt.yticks(np.arange(0, len(corrs)))
    plt.ylabel(axes[0])
    plt.xlabel(axes[1])
    # plt.show()
    plt.savefig(fname=full_path + fname)
    plt.close()

