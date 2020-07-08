import datetime as dt


class Logger:
    def __init__(self, experiment_type, constants, prefix='default_'):
        self.constants = constants
        self.log_version = prefix + '{}_lr_{}_batchsize_{}_trainiters_{}_rowspertrainiter_{}_uuid_{}'.\
            format(experiment_type, '{:1.3f}'.format(constants.learn_rate).replace('.', '_'), constants.batch_size,
                   constants.train_iters, constants.rows_per_train_iter, constants.UUID)

    def log(self, params=[], log_str='', opt_fname_postfix=False):
        if not opt_fname_postfix:
            opt_fname_postfix = self.constants.optimiser.__name__

        fname = './Logs/' + self.log_version
        if opt_fname_postfix != '':
            fname += '_' + opt_fname_postfix
        fname += '.txt'

        prefix = '[{}]'.format(dt.datetime.now())
        if len(params) > 0:
            prefix = prefix + ' ---------- parameters: {}'.format(params)
        full_str = prefix + ' ' + log_str + '\n'
        print('Writing to log:\n{}'.format(full_str))
        with open(fname, 'a') as f:
            f.write(full_str)
