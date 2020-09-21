import nevergrad as ng
import torch
from nevergrad.functions import MultiobjectiveFunction

from Dev.brian_GLIF_model_setup import *
from Dev.setup_data_for_brian import *
from Log import Logger
from eval import calculate_loss
from gf_metric import compute_gamma_factor_for_lists, get_spikes

targets = torch.tensor(targets)


def run_simulation_multiobjective(w, C_m, G, R_I, f_v, f_I, E_L, t_interval=4000*ms):
    restore()

    synapses.w = w
    neurons.f_I = f_I
    neurons.C_m = C_m
    neurons.G = G
    neurons.R_I = R_I*ohm
    neurons.f_v = f_v
    neurons.E_L = E_L*mV

    in_grp.set_spikes(np.reshape(input_indices, (-1,)), np.reshape(input_times*ms, (-1,)))
    spikemon = SpikeMonitor(neurons[:], 'v', record=True)

    run(t_interval)
    print('#spikes in simulation:', spikemon.num_spikes)

    m_spike_times = get_spikes(spikemon)
    t_spike_times = data_util.scale_spike_times(target_spike_times)  # ms to seconds
    gf = compute_gamma_factor_for_lists(m_spike_times, t_spike_times, time=t_interval, delta=1*ms)

    brian_model_spike_train = data_util.convert_brian_spike_train_dict_to_boolean_matrix(spikemon.spike_trains(), t_max=t_interval/ms)
    brian_model_spike_train = torch.tensor(brian_model_spike_train, dtype=torch.float)
    vr_dist = np.float(calculate_loss(brian_model_spike_train, targets, loss_fn='van_rossum_dist', tau_vr=3.0))
    poisson_nll = np.float(calculate_loss(brian_model_spike_train, targets, loss_fn='poisson_nll'))
    return [vr_dist, poisson_nll, gf]


def run_simulation_gamma_factor(w, C_m, G, R_I, f_v, f_I, E_L, t_interval=4000*ms, loss_fn='van_rossum_dist'):
    restore()

    synapses.w = w
    neurons.f_I = f_I
    neurons.C_m = C_m
    neurons.G = G
    neurons.R_I = R_I*ohm
    neurons.f_v = f_v
    neurons.E_L = E_L*mV

    in_grp.set_spikes(np.reshape(input_indices, (-1,)), np.reshape(input_times*ms, (-1,)))
    spikemon = SpikeMonitor(neurons[:], 'v', record=True)

    run(t_interval)
    print('#spikes in simulation:', spikemon.num_spikes)

    if loss_fn == 'gamma_factor':
        m_spike_times = get_spikes(spikemon)
        t_spike_times = data_util.scale_spike_times(target_spike_times)  # ms to seconds
        return compute_gamma_factor_for_lists(m_spike_times, t_spike_times, time=t_interval, delta=1*ms)
    elif loss_fn in ['van_rossum_dist', 'poisson_nll', 'kl_div']:
        brian_model_spike_train = data_util.convert_brian_spike_train_dict_to_boolean_matrix(spikemon.spike_trains(), t_max=t_interval/ms)
        brian_model_spike_train = torch.tensor(brian_model_spike_train, dtype=torch.float)
        return np.float(calculate_loss(brian_model_spike_train, targets, loss_fn=loss_fn, tau_vr=3.0))


N = 12
w_mean = 0.3; w_var = 0.5; rand_ws = (w_mean - w_var) + 2 * w_var * np.random.random((N**2))
instrum = ng.p.Instrumentation(w=ng.p.Array(init=rand_ws).set_bounds(-1., 1.),
                               f_I=ng.p.Array(init=0.4 * np.ones((N,))).set_bounds(0.01, 0.99),
                               C_m=ng.p.Array(init=1.5 * np.ones((N,))).set_bounds(1., 3.),
                               G=ng.p.Array(init=0.8 * np.ones((N,))).set_bounds(0.01, 0.99),
                               R_I=ng.p.Array(init=18. * np.ones((N,))).set_bounds(10., 30.),
                               f_v=ng.p.Array(init=0.14 * np.ones((N,))).set_bounds(0.01, 0.99),
                               E_L=ng.p.Array(init=-65. * np.ones((N,))).set_bounds(-90., -30.))

# TODO: PSO, NGO, CMA(?)
optimizer = ng.optimizers.DE(parametrization=instrum, budget=10)

logger = Logger(log_fname='brian2_network_nevergrad_optimization')
logger.log('setup experiment with the optimizer {}'.format(optimizer.__str__()))

multiobjective_fn = MultiobjectiveFunction(multiobjective_function=run_simulation_multiobjective)
recommendation = optimizer.minimize(multiobjective_fn, verbosity=2)

logger.log('recommendation.value: {}'.format(recommendation.value))

logger.log("Random subset:", optimizer.pareto_front(2, subset="random"))
logger.log("Loss-covering subset:", optimizer.pareto_front(2, subset="loss-covering"))
logger.log("Domain-covering subset:", optimizer.pareto_front(2, subset="domain-covering"))
