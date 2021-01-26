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

    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-06-33-420/plot_losses01-20_20-09-52-007.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-06-33-420/plot_losses01-21_01-21-22-140.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-06-33-420/plot_losses01-21_06-30-31-588.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-06-33-420/plot_losses01-21_11-38-34-092.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-06-33-420/plot_losses01-21_16-48-36-100.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_16-48-38-729/plot_losses01-21_21-55-40-785.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_16-48-38-729/plot_losses01-22_03-04-19-720.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_16-48-38-729/plot_losses01-22_08-17-58-071.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_16-48-38-729/plot_losses01-22_13-35-55-784.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_16-48-38-729/plot_losses01-22_18-58-43-144.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-22_18-58-47-549/plot_losses01-23_00-20-26-553.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-22_18-58-47-549/plot_losses01-23_05-49-59-224.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-22_18-58-47-549/plot_losses01-23_11-25-13-932.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-22_18-58-47-549/plot_losses01-23_17-00-47-813.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-22_18-58-47-549/plot_losses01-23_22-37-25-709.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-23_22-37-29-636/plot_losses01-24_04-11-55-613.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-23_22-37-29-636/plot_losses01-24_09-52-40-855.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-23_22-37-29-636/plot_losses01-24_15-35-33-828.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-23_22-37-29-636/plot_losses01-24_21-16-42-023.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-23_22-37-29-636/plot_losses01-25_03-12-01-776.pt']

    # load_path = '/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/'

    # fname = load_paths[0].split('/')[-2]
    # fname = fname.split('.pt')[0].replace('.', '_')
    # save_fname = 'export_{}.eps'.format(fname)
    save_fname = 'export_SGD_vrd.eps'
    custom_title = 'Average loss, SGD, $\\alpha=0.05$, van Rossum distance'

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
