from Models.GLIF import GLIF
from experiments import draw_from_uniform, zip_dicts, poisson_input

free_parameters = {'C_m', 'G', 'E_L', 'delta_theta_s', 'b_s', 'f_v', 'delta_V', 'f_I', 'I_A', 'b_v', 'a_v', 'theta_inf', 'R_I'}

num_neurons = 12
params_model = draw_from_uniform(free_parameters, GLIF.parameter_init_intervals, num_neurons)
params_model = zip_dicts(params_model, {})

sut = GLIF(parameters=params_model, N=num_neurons)

t_inputs = poisson_input(rate=10., t=4000, N=sut.v.shape[0])

_, sut_outs = sut(t_inputs)

print('sum of sut_outs: {}'.format(sut_outs.sum()))
