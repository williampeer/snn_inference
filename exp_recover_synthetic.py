import data_util
import main
from Models.LIF import LIF_pulse_syn, LIF
from experiments import *
from model_util import generate_model_data
from plot import *
import exp_suite as experim, Log, Constants as C

torch.autograd.set_detect_anomaly(True)

# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = 'cpu'
verbose = True
# ---------------------------------------


# def dev_exp():
#     Log.log([main.constants], 'Starting dev exp. with the aforementioend alpha, train iters, exp N, batch size, tau van rossum, input coeff.')
#     initial_parameters = { 'tau_m': 5.5, 'tau_g': 3.0, 'v_rest': -65.0, 'N': 12, 'w_mean': 0.25, 'w_var': 0.5 }
#     Log.log('', 'initial model parameters: {}'.format(initial_parameters))
#     experim.recover_model_parameters(model_class=LIF_conductance_syn,
#                                      initial_model_parameters=initial_parameters,
#                                      initial_gen_parameters=initial_parameters)


def recover_parameter_LIF_pop_helper(gen_mean=0.8, gen_var=0., gen_tau_m=6.5, gen_tau_g=5.0, gen_v_rest=-65.,
                                     model_mean=0.8, model_var=0., m_tau_m=6.5, m_tau_g=5.0, m_v_rest=-65.,
                                     learn_rate=5e-03, train_iters=40, exp_N=4, batch_size=60,
                                     tau_van_rossum=torch.tensor(4.0)):
    # gen_model = LIF_pulse_syn(device, tau_m=gen_tau_m, v_rest=gen_v_rest, N=3, w_mean=gen_mean, w_var=gen_var)
    gen_model = LIF(device, tau_m=gen_tau_m, tau_g=gen_tau_g, v_rest=gen_v_rest, N=3, w_mean=gen_mean, w_var=gen_var)


def divergence_test_exp():
    print('==============RUNNING divergence_test_exp()==============')
    recover_parameter_LIF_pop_helper(train_iters=10, exp_N=10, learn_rate=0.1)


def recover_pop_params_different_taus():
    print('==============RUNNING recover_pop_params_different_taus()==============')
    recover_parameter_LIF_pop_helper(train_iters=20, exp_N=5, learn_rate=0.05,
                                     gen_tau_m=5.0, m_tau_m=6.5)

    recover_parameter_LIF_pop_helper(train_iters=20, exp_N=5, learn_rate=0.05,
                                     gen_tau_g=2.5, m_tau_m=5.5)

    recover_parameter_LIF_pop_helper(train_iters=20, exp_N=5, learn_rate=0.05,
                                     gen_mean=0.8, gen_var=0.2, model_mean=0.5, model_var=0.15)


def recover_pop_params_different_rest_potentials():
    print('=============RUNNINg recover_pop_params_different_rest_potentials()============')
    recover_parameter_LIF_pop_helper(train_iters=10, exp_N=5, learn_rate=0.1,
                                     gen_v_rest=-52.5, m_v_rest=-65.0)

    recover_parameter_LIF_pop_helper(train_iters=10, exp_N=5, learn_rate=0.1,
                                     gen_v_rest=-72.5, m_v_rest=-65.0)

    recover_parameter_LIF_pop_helper(train_iters=10, exp_N=5, learn_rate=0.1,
                                     gen_v_rest=-70.0, m_v_rest=-55.0)
