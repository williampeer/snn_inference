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
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_losses01-20_21-09-57-287.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_losses01-21_03-29-46-816.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_losses01-21_09-44-53-072.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_losses01-21_15-58-01-304.pt']  # outlier
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_losses01-21_22-08-02-864.pt']

    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_22-08-07-832/plot_losses01-22_04-15-29-139.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_22-08-07-832/plot_losses01-22_10-25-39-631.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_22-08-07-832/plot_losses01-22_16-40-29-469.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_22-08-07-832/plot_losses01-22_22-57-36-228.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_22-08-07-832/plot_losses01-23_05-11-32-896.pt']

    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-23_05-11-39-091/plot_losses01-23_11-34-51-713.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-23_05-11-39-091/plot_losses01-23_18-02-56-473.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-23_05-11-39-091/plot_losses01-24_00-35-45-591.pt']  # outlier
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-23_05-11-39-091/plot_losses01-24_07-08-39-126.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-23_05-11-39-091/plot_losses01-24_13-44-17-821.pt']  # outlier

    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-24_13-44-20-353/plot_losses01-24_20-19-01-392.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-24_13-44-20-353/plot_losses01-25_02-42-34-284.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-24_13-44-20-353/plot_losses01-25_08-42-15-266.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-24_13-44-20-353/plot_losses01-25_15-37-48-875.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-24_13-44-20-353/plot_losses01-25_23-01-42-059.pt']

    # load_path = '/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/'


    # fname = load_paths[0].split('/')[-2]
    # fname = fname.split('.pt')[0].replace('.', '_')
    # save_fname = 'export_{}.eps'.format(fname)
    save_fname = 'export_Adam_frd.eps'
    custom_title = 'Average loss, Adam, $\\alpha=0.05$, firing rate distance'

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
