import brian2modelfitting as b2f

import data_util

model = '''
dv/dt = (G * (E_L - v) + R_I * (I_ext)) / C_m : volt
dtheta_v/dt = a_v * (v - E_L) - b_v * (theta_v - theta_inf)
dtheta_s/dt = - b_s * theta_s
# dI_syn/dt = -f_I * I_syn : amp
f_I : 1
I_A : 1 (constant)
E_L : voltage (constant)
R_I : ohm (constant)
b_s : 1 (constant)
a_v : 1 (constant)
b_v : 1 (constant)
delta_V : 1 (constant)
theta_inf : voltage (constant)
'''

reset = '''
v = E_L + f_v * (v - E_L) - delta_V
theta_s = theta_s - b_s * theta_s + delta_theta_s
# theta_v = theta_v
delta_theta_s : voltage (constant)
f_v : 1 (constant)
# I_syn = I_syn - f_I * I_syn + I_A
'''

init_params = '''
G = 0.7 * ohm
E_L = -65. * mV
v = -65. * mV
theta_v = 1 * mV
theta_s = 30. * mV
# I_syn = 0. * amp
C_m = 1. * farad
delta_theta_s = 30.
b_s = 0.3
b_v = 0.5
a_v = 0.5
f_v = 0.15
delta_V = 12. * mV
f_I = 0.3
theta_inf = -20. * mV
'''

fitter = b2f.fitter.SpikeFitter(model, input, output, dt=0.1*b2f.ms, reset=reset,
                                threshold = 'v>30*mV', input_var='I_ext : amp', refractory=False,
                                n_samples=100, method='euler', param_init=init_params,
                                penalty=None, use_units=True)

results, error = fitter.fit(n_rounds=2, optimizer=b2f.SkoptOptimizer('ET'),
                            metric=b2f.GammaFactor(dt=0.1*b2f.ms, delta=1*b2f.ms))
