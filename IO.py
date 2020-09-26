import torch
import os
from datetime import datetime

import data_util

PATH = './saved/'
PLOT_PATH = 'plot_data/'
fname_ext = '.pt'


def makedir_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def save_entire_model(model, uuid, fname='test_model'):
    makedir_if_not_exists(PATH + uuid)

    torch.save(model, PATH+uuid+'/'+fname+fname_ext)


def save_model_params(model, fname='test_model_params'):
    full_path = data_util.prefix + data_util.path
    makedir_if_not_exists(full_path)

    torch.save(model.state_dict(), full_path+'/'+fname+fname_ext)


def save(model, loss, uuid, fname='test_exp_dict'):
    makedir_if_not_exists(PATH + uuid)

    torch.save({
        'model': model,
        'loss': loss
    }, PATH+uuid+'/'+fname+fname_ext)


def save_plot_data(data, uuid, plot_fn='unknown', fname=False):
    makedir_if_not_exists(PATH+PLOT_PATH+uuid)

    if not fname:
        fname = plot_fn + dt_descriptor()
    torch.save({
        'plot_data': data,
        'plot_fn': plot_fn
    }, PATH+PLOT_PATH+uuid+'/'+fname+fname_ext)


def dt_descriptor():
    return datetime.utcnow().strftime('%m-%d_%H-%M-%S-%f')[:-3]
