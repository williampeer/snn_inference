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

    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-07-44-795/plot_losses01-20_20-33-14-308.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-07-44-795/plot_losses01-21_01-56-58-028.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-07-44-795/plot_losses01-21_07-23-22-825.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-07-44-795/plot_losses01-21_12-49-03-301.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-07-44-795/plot_losses01-21_18-12-54-080.pt']

    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_18-12-56-967/plot_losses01-21_23-36-09-468.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_18-12-56-967/plot_losses01-22_05-00-39-102.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_18-12-56-967/plot_losses01-22_10-28-53-700.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_18-12-56-967/plot_losses01-22_16-01-44-776.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_18-12-56-967/plot_losses01-22_21-37-37-987.pt']

    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-22_21-37-40-455/plot_losses01-23_03-16-04-280.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-22_21-37-40-455/plot_losses01-23_09-01-40-005.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-22_21-37-40-455/plot_losses01-23_14-51-07-075.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-22_21-37-40-455/plot_losses01-23_20-40-41-137.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-22_21-37-40-455/plot_losses01-24_02-32-38-668.pt']

    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-24_02-32-43-141/plot_losses01-24_08-24-47-528.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-24_02-32-43-141/plot_losses01-24_14-22-54-644.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-24_02-32-43-141/plot_losses01-24_20-21-57-088.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-24_02-32-43-141/plot_losses01-25_02-27-32-631.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-24_02-32-43-141/plot_losses01-25_08-45-34-237.pt']

    # load_path = '/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/'

    # fname = load_paths[0].split('/')[-2]
    # fname = fname.split('.pt')[0].replace('.', '_')
    # save_fname = 'export_{}.eps'.format(fname)
    save_fname = 'export_Adam_vrd.eps'
    custom_title = 'Average loss, Adam, $\\alpha=0.05$, van Rossum distance'

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
