import nevergrad as ng
import torch

from Dev.brian_GLIF_model_setup import *
from Dev.setup_data_for_brian import *
from Log import Logger
from eval import calculate_loss

targets = torch.tensor(targets)


def run_simulation(w, C_m, G, R_I, f_v, f_I, E_L, t_interval=4000*ms):
    restore()

    synapses.w = w
    synapses.f_I = f_I

    neurons.C_m = C_m
    neurons.G = G
    neurons.R_I = R_I*ohm
    neurons.f_v = f_v
    neurons.E_L = E_L*mV
    # TODO: set further params

    synapses.I_syn = 10 * mA  # TODO: input

    # TODO: multi-objective optimization(?)
    # losses = []
    # for i in range(t_interval/batch_size):
    #     run(batch_size)
    #     print('num spikes:', spikemon.num_spikes)
    #     cur_loss = calculate_loss(spikemon.spike_trains().values(), targets, loss_fn='poisson_nll').numpy()
    #     losses.append(cur_loss)
    # return np.mean(losses)

    run(t_interval)
    print('num spikes:', spikemon.num_spikes)
    brian_model_spike_train = data_util.convert_brian_spike_train_dict_to_boolean_matrix(spikemon.spike_trains(), t_max=t_interval/ms)
    brian_model_spike_train = torch.tensor(brian_model_spike_train)
    # return np.float(calculate_loss(brian_model_spike_train, targets, loss_fn='van_rossum_dist', tau_vr=3.0))
    return np.float(calculate_loss(brian_model_spike_train, targets, loss_fn='poisson_nll'))


N = 12
instrum = ng.p.Instrumentation(w=ng.p.Array(shape=(N**2,)).set_bounds(-1., 1.),
                               f_I=ng.p.Array(init=0.4 * np.ones((N**2,))).set_bounds(0.01, 0.99),
                               C_m=ng.p.Array(init=1.5 * np.ones((N,))).set_bounds(1., 3.),
                               G=ng.p.Array(init=0.8 * np.ones((N,))).set_bounds(0.01, 0.99),
                               R_I=ng.p.Array(init=18. * np.ones((N,))).set_bounds(10., 30.),
                               f_v=ng.p.Array(init=0.14 * np.ones((N,))).set_bounds(0.01, 0.99),
                               E_L=ng.p.Array(init=-65. * np.ones((N,))).set_bounds(-90., -30.))

optimizer = ng.optimizers.DE(parametrization=instrum, budget=10)
logger = Logger(log_fname='brian2_network_nevergrad_optimization')
logger.log('setup experiment with the optimizer {}'.format(optimizer.__str__()))
recommendation = optimizer.minimize(run_simulation)  # best value

logger.log('recommendation.value: {}'.format(recommendation.value))
