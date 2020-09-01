import enum

from torch import optim

import IO


class Constants:
    def __init__(self, data_bin_size, target_bin_size, learn_rate, train_iters, N_exp, batch_size, tau_van_rossum,
                 initial_poisson_rate, rows_per_train_iter, optimiser, loss_fn, data_set, evaluate_step, fitted_model_path):
        self.data_bin_size = int(data_bin_size)
        self.target_bin_size = int(target_bin_size)
        self.learn_rate = float(learn_rate)
        self.train_iters = int(train_iters)
        self.N_exp = int(N_exp)
        self.batch_size = int(batch_size)
        self.tau_van_rossum = float(tau_van_rossum)
        self.initial_poisson_rate = float(initial_poisson_rate)
        self.rows_per_train_iter = int(rows_per_train_iter)
        self.loss_fn = loss_fn
        self.data_set = data_set
        self.evaluate_step = evaluate_step
        self.fitted_model_path = fitted_model_path

        # self.UUID = uuid.uuid4().__str__()
        self.UUID = IO.dt_descriptor()

        self.optimiser = None
        if optimiser == 'SGD':
            self.optimiser = optim.SGD
        elif optimiser == 'Adam':
            self.optimiser = optim.Adam
        else:
            print('Optimiser not supported. Please use either SGD or Adam.')

    def __str__(self):
        return 'data_bin_size: {}, target_bin_size: {}, learn_rate: {}, train_iters: {}, N_exp: {}, batch_size: {},' \
               'tau_van_rossum: {}, initial_poisson_rate: {}, rows_per_train_iter: {}, ' \
               'optimiser: {}, loss_fn: {}, data_set: {}, evaluate_step: {}, fitted_model_path: {}'.\
            format(self.data_bin_size, self.target_bin_size, self.learn_rate, self.train_iters, self.N_exp,
                   self.batch_size, self.tau_van_rossum, self.initial_poisson_rate, self.rows_per_train_iter,
                   self.optimiser, self.loss_fn, self.data_set, self.evaluate_step, self.fitted_model_path)


class ExperimentType(enum.Enum):
    DataDriven = 1
    Synthetic = 2
    SanityCheck = 3
    RetrieveFitted = 4
