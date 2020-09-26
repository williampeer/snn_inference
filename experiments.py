import torch
import numpy as np

from model_util import generate_model_data

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def draw_from_uniform(parameters, parameter_intervals, N):
    new_dict = {}
    for i, k in enumerate(parameters):
        low = parameter_intervals[k][0]; high = parameter_intervals[k][1]
        new_dict[k] = torch.FloatTensor(np.random.uniform(low, high, N))
    return new_dict


def randomise_parameters(initial_parameters, coeff=torch.tensor(0.5), N_dim=None):
    res = initial_parameters.copy()
    for key in initial_parameters.keys():
        # rand_sign = round(torch.randn())*2 -1
        if N_dim is not None:
            rand_sign = 2 * torch.randint(0, 1, (1,))[0] - 1
            res[key] = res[key] + rand_sign * coeff * torch.randn((int(N_dim),)) * res[key]
        else:
            rand_sign = 2 * torch.randint(0, 1, (1,))[0] - 1
            res[key] = res[key] + rand_sign * coeff * torch.randn((1,))[0] * res[key]

    return res


def zip_dicts(a, b):
    res = a.copy()
    for key in b.keys():
        if key in a.keys():
            res[key].append(b[key])
        else:
            res[key] = b[key]
    return res


def poisson_input(rate, t, N):
    return torch.poisson(rate * torch.ones((int(t), N)))  # t x N


def generate_synthetic_data(gen_model, poisson_rate, t):
    gen_input = poisson_input(rate=poisson_rate, t=t, N=gen_model.N)
    _, gen_spiketrain = generate_model_data(model=gen_model, inputs=gen_input)
    # for gen spiketrain this may be thresholded to binary values:
    gen_spiketrain = torch.round(gen_spiketrain)
    del gen_input

    return gen_spiketrain.clone().detach()


def train_test_split(data, train_test_split_factor=0.85):
    assert data.shape[0] > data.shape[1], "assert row as timestep matrix. data.size: {}".format(data.size)
    splice_index = int(train_test_split_factor * data.size(0))
    train_data = data[:splice_index]
    test_data = data[splice_index:]
    return train_data, test_data


def train_val_test_split(data, train_factor=0.6, val_factor=0.2, test_factor=0.2):
    f_sum = (train_factor + val_factor + test_factor)
    assert f_sum == 1, "the factors must sum to one. sum:{}".format(f_sum)

    splice_train_index = int(train_factor*data.size(0))
    splice_val_index = int(val_factor*data.size(0))

    train_data = data[:splice_train_index]
    val_data = data[splice_train_index:splice_val_index]
    test_data = data[(splice_train_index+splice_val_index):]

    return train_data, val_data, test_data
