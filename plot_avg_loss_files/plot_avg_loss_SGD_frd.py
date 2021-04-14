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

    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_losses01-20_21-10-55-763.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_losses01-21_03-37-03-791.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_losses01-21_09-57-28-930.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_losses01-21_16-14-17-948.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-04-49-336/plot_losses01-21_22-30-26-503.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_22-30-30-455/plot_losses01-22_04-44-44-131.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_22-30-30-455/plot_losses01-22_10-59-24-176.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_22-30-30-455/plot_losses01-22_17-20-16-670.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_22-30-30-455/plot_losses01-22_23-40-37-966.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-21_22-30-30-455/plot_losses01-23_06-07-24-063.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-23_06-07-29-245/plot_losses01-23_12-39-51-927.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-23_06-07-29-245/plot_losses01-23_19-16-07-606.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-23_06-07-29-245/plot_losses01-24_01-53-00-859.pt']
    # load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-23_06-07-29-245/plot_losses01-24_08-32-42-085.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-23_06-07-29-245/plot_losses01-24_15-13-57-920.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-24_15-14-04-415/plot_losses01-24_21-54-48-238.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-24_15-14-04-415/plot_losses01-25_04-13-22-256.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-24_15-14-04-415/plot_losses01-25_10-35-13-260.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-24_15-14-04-415/plot_losses01-25_17-43-20-750.pt']
    load_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-24_15-14-04-415/plot_losses01-26_01-24-15-961.pt']

    # load_path = '/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/'

    # fname = load_paths[0].split('/')[-2]
    # fname = fname.split('.pt')[0].replace('.', '_')
    # save_fname = 'export_{}.eps'.format(fname)
    save_fname = 'export_SGD_frd.eps'
    custom_title = 'Average loss, SGD, $\\alpha=0.05$, firing rate distance'

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
