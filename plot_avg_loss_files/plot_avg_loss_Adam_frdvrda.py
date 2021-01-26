import sys

import torch
import numpy as np

import Log
import plot
import stats
from Constants import ExperimentType
from Models.LIF import LIF


def main(argv):
    print('Argument List:', str(argv))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]

    load_paths = []

    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_14-52-28-339/plot_losses01-21_18-53-40-193.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_14-52-28-339/plot_losses01-21_22-53-46-252.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_14-52-28-339/plot_losses01-22_03-01-28-450.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_14-52-28-339/plot_losses01-22_07-04-17-404.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_14-52-28-339/plot_losses01-22_11-04-07-996.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-22_11-04-09-239/plot_losses01-22_15-11-53-955.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-22_11-04-09-239/plot_losses01-22_19-17-14-533.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-22_11-04-09-239/plot_losses01-22_23-23-11-880.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-22_11-04-09-239/plot_losses01-23_03-28-26-881.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-22_11-04-09-239/plot_losses01-23_07-35-11-342.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-23_07-35-14-176/plot_losses01-23_11-48-04-525.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-23_07-35-14-176/plot_losses01-23_16-05-00-576.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-23_07-35-14-176/plot_losses01-23_20-24-03-352.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-23_07-35-14-176/plot_losses01-24_00-45-13-716.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-23_07-35-14-176/plot_losses01-24_05-06-46-277.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-24_05-06-49-443/plot_losses01-24_09-25-45-365.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-24_05-06-49-443/plot_losses01-24_13-46-56-752.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-24_05-06-49-443/plot_losses01-24_18-09-25-649.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-24_05-06-49-443/plot_losses01-24_22-27-02-104.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-24_05-06-49-443/plot_losses01-25_02-34-47-679.pt']

    # load_path = '/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/'

    # fname = load_paths[0].split('/')[-2]
    # fname = fname.split('.pt')[0].replace('.', '_')
    # save_fname = 'export_{}.eps'.format(fname)
    save_fname = 'export_Adam_frdvrda.eps'
    custom_title = 'Average loss, Adam, $\\alpha=0.05$, composite adaptive distance'

    train_losses = []; test_losses = []
    for lp in load_paths:
        data = torch.load(lp)
        print('Loaded saved plot data.')

        plot_data = data['plot_data']
        # plot_fn = data['plot_fn']
        cur_train_loss = plot_data['training_loss']
        train_losses.append(cur_train_loss)
        cur_test_loss = plot_data['test_loss']
        test_losses.append(cur_test_loss)

    avg_train_loss = np.mean(np.asarray(train_losses), axis=1)
    std_train_loss = np.std(np.asarray(train_losses), axis=1)
    avg_test_loss = np.mean(np.asarray(test_losses), axis=1)
    std_test_loss = np.std(np.asarray(test_losses), axis=1)

    plot.plot_avg_losses(avg_train_loss, std_train_loss, avg_test_loss, std_test_loss, uuid='export',
                         custom_title=custom_title, fname=save_fname)


if __name__ == "__main__":
    main(sys.argv[1:])
