import datetime as dt


class Logger:
    def __init__(self, log_fname):
        self.fname = './Logs/' + log_fname + '.txt'

    def log(self, log_str='', params=[]):
        prefix = '[{}]'.format(dt.datetime.now())
        if len(params) > 0:
            prefix = prefix + ' ---------- parameters: {}'.format(params)
        full_str = prefix + ' ' + log_str + '\n'
        print('Writing to log:\n{}'.format(full_str))
        with open(self.fname, 'a') as f:
            f.write(full_str)
