import data_util

import brian2modelfitting as b2f
from brian2modelfitting import *
from brian2 import *

from Log import Logger
from experiments import poisson_input

from Dev.setup_data_for_brian import *

# logger = Logger('brian2_fitting_playbook_' + output_fname.replace('.mat', ''))
logger = Logger('brian2_fitting_playbook_poisson_input_' + output_fname.replace('.mat', ''))

for neuron_index in range(0, 12):
    # neuron_index = 0
    # current_first_inputs = np.reshape(first_inputs[:, neuron_index], (-1, 1))
    current_inputs = poisson_input(0.6, t=time_interval, N=1).numpy()
    current_outputs = np.reshape(targets[:, neuron_index], (-1, 1))

    # E_L = -60.; b_s = 0.4; b_v = 0.5; a_v = 0.5; delta_theta_s = 25.; delta_V = 12.; theta_innf = -25.
    tau = 1*ms
    GLIF_eqs = '''
    dv/dt = ((G * (E_L - v) + R_I * I) / C_m)/tau : 1
    dtheta_v/dt = (a_v * (v - E_L) - b_v * (theta_v - theta_innf))/tau : 1
    dtheta_s/dt = (- b_s * theta_s)/tau : 1
    C_m : 1 (constant)
    G : 1 (constant)
    R_I : 1 (constant)
    f_v : 1 (constant)
    E_L : 1 (constant)
    b_s : 1 (constant)
    b_v : 1 (constant)
    a_v : 1 (constant)
    delta_theta_s : 1 (constant)
    delta_V : 1 (constant)
    theta_innf : 1 (constant)
    '''
    reset = '''
    v = E_L + f_v * (v - E_L) - delta_V
    theta_s = theta_s - b_s * theta_s + delta_theta_s
    '''

    init_params = {'C_m': 1.5, 'G': 0.8, 'R_I': 18., 'f_v': 0.14,
                   'delta_theta_s': 25., 'b_s': 0.4, 'delta_V': 12.,
                   'b_v': 0.5, 'a_v': 0.5, 'theta_innf': -25.}
    sf = b2f.fitter.SpikeFitter(GLIF_eqs, input=current_inputs, output=current_outputs, dt=0.5 * ms, reset=reset,
                                threshold='v>30', refractory=False,
                                n_samples=100, method='euler', param_init=init_params,
                                penalty=None, use_units=True)
    attr_fitter = ['dt', 'simulator', 'parameter_names', 'n_traces',
                   'duration', 'n_neurons', 'n_samples', 'method', 'threshold',
                   'reset', 'refractory', 'input', 'output', 'output_var',
                   'best_params', 'input_traces', 'model', 'optimizer',
                   'metric']
    for attr in attr_fitter:
        assert hasattr(sf, attr)

    results, error = sf.fit(n_rounds=1,
                            # optimizer=SkoptOptimizer(),
                            # optimizer=SkoptOptimizer(method='GP', acq_func='LCB'),
                            optimizer=NevergradOptimizer(),
                            metric=GammaFactor(delta=100*ms, time=time_interval * ms),
                            C_m=[1.0, 2.0], G=[0.3, 1.0], R_I=[15., 30.], f_v=[0.01, 0.99],
                            delta_V=[1., 65.],
                            E_L=[-65., -52.],
                            b_v=[0.01, 0.99], a_v=[0.01, 0.99],
                            b_s=[0.1, 0.9], delta_theta_s=[1., 30.], theta_innf=[-40., -1.])
    logger.log(parameters=results, log_str='Fitted neuron #{}, error:{}'.format(neuron_index, error))
