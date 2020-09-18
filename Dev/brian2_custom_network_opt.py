import nevergrad as ng
import torch

from Dev.brian_GLIF_model_setup import *
from Dev.setup_data_for_brian import *
from Log import Logger
from eval import calculate_loss

targets = torch.tensor(targets)


def run_simulation(E_L, w, R_I, t_interval=4000*ms):
    restore()

    neurons.E_L = E_L*mV
    neurons.R_I = R_I*ohm
    synapses.w = w
    # TODO: set further params

    synapses.I_syn = 10 * mA  # TODO: input

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
instrum = ng.p.Instrumentation(E_L=ng.p.Array(init=-65. * np.ones((N,))).set_bounds(-90., -30.),
                               w=ng.p.Array(shape=(N**2,)).set_bounds(-1., 1.),
                               R_I=ng.p.Array(init=18. * np.ones((N,))).set_bounds(10., 30.))

optimizer = ng.optimizers.DE(parametrization=instrum, budget=61)
logger = Logger(log_fname='brian2_network_nevergrad_optimization')
logger.log('setup experiment with the optimizer {}'.format(optimizer.__str__()))
recommendation = optimizer.minimize(run_simulation)  # best value

logger.log('recommendation.value: {}'.format(recommendation.value))
