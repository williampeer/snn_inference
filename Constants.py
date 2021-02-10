import enum

from torch import optim

import IO


class Constants:
    def __init__(self, learn_rate, train_iters, N_exp, batch_size, tau_van_rossum,
                 initial_poisson_rate, rows_per_train_iter, optimiser, loss_fn, evaluate_step,
                 plot_flag=True, start_seed=0, target_fname=None, exp_type_str=None):

        self.learn_rate = float(learn_rate)
        self.train_iters = int(train_iters)
        self.N_exp = int(N_exp)
        self.batch_size = int(batch_size)
        self.tau_van_rossum = float(tau_van_rossum)
        self.initial_poisson_rate = float(initial_poisson_rate)
        self.rows_per_train_iter = int(rows_per_train_iter)
        self.loss_fn = loss_fn
        self.evaluate_step = evaluate_step
        self.plot_flag = plot_flag
        self.start_seed = start_seed
        try:
            self.EXP_TYPE = ExperimentType[exp_type_str]
        except:
            raise NotImplementedError('ExperimentType not found.')

        self.UUID = IO.dt_descriptor()

        self.optimiser = None
        if optimiser == 'SGD':
            self.optimiser = optim.SGD
        elif optimiser == 'Adam':
            self.optimiser = optim.Adam
        else:
            print('Optimiser not supported. Please use either SGD or Adam.')

    def __str__(self):
        return 'learn_rate: {}, train_iters: {}, N_exp: {}, batch_size: {},' \
               'tau_van_rossum: {}, initial_poisson_rate: {}, rows_per_train_iter: {}, ' \
               'optimiser: {}, loss_fn: {}, evaluate_step: {},' \
               'plot_flag: {}, start_seed: {}, EXP_TYPE: {}'.\
            format(self.learn_rate, self.train_iters, self.N_exp,
                   self.batch_size, self.tau_van_rossum, self.initial_poisson_rate, self.rows_per_train_iter,
                   self.optimiser, self.loss_fn, self.evaluate_step,
                   self.plot_flag, self.start_seed, self.EXP_TYPE)


class ExperimentType(enum.Enum):
    DataDriven = 1
    Synthetic = 2
    SanityCheck = 3
    RetrieveFitted = 4
