import sys

import torch

import plot


def main(argv):
    print('Argument List:', str(argv))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]

    load_path = None
    # load_path = './saved/plot_data/06-24_10-19-53-648/plot_spiketrains_side_by_side06-24_10-20-26-872.pt'

    for i, opt in enumerate(opts):
        if opt == '-h':
            print('load_and_export_plot_data.py -p <path>')
            sys.exit()
        elif opt in ("-p", "--path"):
            load_path = args[i]

    if load_path is None:
        print('No path to load model from specified.')
        sys.exit(1)

    data = torch.load(load_path)
    print('Loaded saved plot data.')

    plot_data = data['plot_data']
    plot_fn = data['plot_fn']

    fname = load_path.split('/')[-1]
    fname = fname.split('.pt')[0]
    save_fname = 'export_{}.eps'.format(fname)
    print('Saving to fname: {}'.format(save_fname))
    if plot_fn == 'plot_spiketrains_side_by_side':
        plot.plot_spiketrains_side_by_side(plot_data['model_spikes'], plot_data['target_spikes'], 'export',
                                           plot_data['exp_type'], plot_data['title'], save_fname, export=True)


if __name__ == "__main__":
    main(sys.argv[1:])
