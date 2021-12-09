import sys

import torch

import IO
import model_util
from Models.microGIF import microGIF
from Test.parameter_landscape_test import plot_param_landscape

A_coeffs = [torch.randn((4,))]
phase_shifts = [torch.rand((4,))]
input_types = [1, 1, 1, 1]
t = 1200
num_steps = 100

target_timestamp_mesoGIF = '12-09_14-56-20-319'
target_timestamp_microGIF = '12-09_14-56-17-312'
target_timestamps = [target_timestamp_mesoGIF, target_timestamp_microGIF]
for str_tt in target_timestamps:
    fname = 'snn_model_target_GD_test'
    load_data = torch.load('./Test/' + IO.PATH + microGIF.__name__ + '/' + str_tt + '/' + fname + IO.fname_ext)
    snn_target = load_data['model']
    # params_model = experiments.draw_from_uniform(GLIF.parameter_init_intervals, N=4)
    # snn_target = TargetModelsBestEffort.glif(random_seed=42, N=4)
    # current_inputs = experiments.generate_composite_input_of_white_noise_modulated_sine_waves(t, A_coeffs, phase_shifts, input_types)
    white_noise = torch.rand((1200, snn_target.N))
    current_inputs = white_noise
    target_spike_probs, target_spikes, target_vs = model_util.feed_inputs_sequentially_return_args(snn_target, current_inputs.clone().detach())

    # other_parameters = experiments.draw_from_uniform(microGIF.parameter_init_intervals, N=snn_target.N)
    other_parameters = snn_target.get_parameters()
    other_parameters['N'] = snn_target.N
    # other_parameters = snn_target.parameters()
    # free_parameters = ['w', 'E_L', 'tau_m', 'tau_s', 'tau_theta', 'J_theta', 'c', 'Delta_u']
    plot_param_landscape(microGIF, [-5., 25.], [2., 20.], 'E_L', 'tau_m', other_parameters, target_spikes, num_steps=num_steps, inputs=current_inputs.clone().detach(), fname_addition='white_noise', GIF_flag=True)
    plot_param_landscape(microGIF, [2., 20.], [1., 20.], 'tau_m', 'tau_s', other_parameters, target_spikes, num_steps=num_steps, inputs=current_inputs.clone().detach(), fname_addition='white_noise', GIF_flag=True)
    plot_param_landscape(microGIF, [800., 1500.], [0.1, 2.], 'tau_theta', 'J_theta', other_parameters, target_spikes, num_steps=num_steps, inputs=current_inputs.clone().detach(), fname_addition='white_noise', GIF_flag=True)
    plot_param_landscape(microGIF, [0.01, 1.0], [1., 20.0], 'c', 'Delta_u', other_parameters, target_spikes, num_steps=num_steps, inputs=current_inputs.clone().detach(), fname_addition='white_noise', GIF_flag=True)

sys.exit(0)
