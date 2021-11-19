import torch

import PDF_metrics
import model_util
import plot


def plot_param_landscape(model_class, p1_interval, p2_interval, p1_name, p2_name, other_parameters, target_spikes,  num_steps, inputs):
    all_parameters = other_parameters

    num_steps_i = num_steps; num_steps_j = num_steps
    p1_step_size = (p1_interval[1] - p1_interval[0])/num_steps_i
    p2_step_size = (p2_interval[1] - p2_interval[0])/num_steps_j
    p1s = []
    p2s = []
    losses = []
    avg_rates = []
    for i_step in range(num_steps_i):
        for j_step in range(num_steps_j):
            cur_p1 = p1_interval[0] + i_step*p1_step_size
            cur_p2 = p2_interval[0] + j_step*p2_step_size
            all_parameters[p1_name] = cur_p1
            all_parameters[p2_name] = cur_p2
            snn = model_class(all_parameters)

            # current_inputs = 0
            spike_probs, spikes, vs = model_util.feed_inputs_sequentially_return_args(snn, inputs)

            loss = PDF_metrics.poisson_nll(spike_probabilities=spike_probs, target_spikes=target_spikes, bin_size=100).clone().detach().numpy()
            losses.append(loss)
            mean_model_rate = spikes.sum(dim=0) * 1000. / spikes.shape[0]  # Hz
            mean_model_rate = torch.mean(mean_model_rate).detach().numpy()
            avg_rates.append(mean_model_rate)
            p1s.append(cur_p1)
            p2s.append(cur_p2)

    plot.plot_parameter_landscape(p1s, p2s, p1_name, p2_name, summary_statistic=losses, statistic_name='loss',
                                  exp_type='param_landscape', uuid='test', fname='test_landscape_{}_{}_losses.png'.format(p1_name, p2_name))
    plot.plot_parameter_landscape(p1s, p2s, p1_name, p2_name, summary_statistic=avg_rates, statistic_name='rate',
                                  exp_type='param_landscape', uuid='test', fname='test_landscape_{}_{}_rates.png'.format(p1_name, p2_name))
