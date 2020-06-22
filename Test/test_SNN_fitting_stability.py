import torch

from Models import LIF, Izhikevich, BaselineSNN
from experiments import poisson_input, train_test_split
from fit import fit_mini_batches
from model_util import generate_model_data
from plot import plot_neuron, plot_losses


def test_stability_with_matching_configurations_deprecated(model, gen_model, input_coefficient, rate_factor, tau_vr, learn_rate):
    t=4000

    poisson_rate = torch.tensor(rate_factor)
    gen_inputs = input_coefficient * poisson_input(poisson_rate, t, gen_model.N)
    gen_membrane_potentials, gen_spiketrain = generate_model_data(model=gen_model, inputs=gen_inputs)

    optims = [torch.optim.SGD(list(model.parameters()), lr=learn_rate),
              torch.optim.SGD([poisson_rate], lr=learn_rate)]
    for neur_ind in range(gen_membrane_potentials.shape[1]):
        plot_neuron(gen_membrane_potentials[:, neur_ind].data, title='Generative neuron model #{}'.format(neur_ind),
                    fname_ext='_test_1_neuron_{}'.format(neur_ind))

    avg_batch_loss = fit_mini_batches(model=model, inputs=gen_inputs,
                                      target_spiketrain=gen_spiketrain, tau_van_rossum=tau_vr,
                                      current_rate=poisson_rate, batch_size=500, uuid='test_SNN_fitting_stability_deprecated',
                                      optimisers=optims)

    for param_i, param in enumerate(list(model.parameters())):
        # print('parameter #{}: {}'.format(param_i, param))
        assert param.grad is not None, "gradient was none. param #{}, \nparam: {}\nparam.grad: {}" \
            .format(param_i, param, param.grad)
        assert torch.abs(param.grad.sum()) > 1e-08, "gradients should not be zero. param #{}, " \
                                                    "\nparam: {}\nparam.grad: {}".format(param_i, param, param.grad)

    return avg_batch_loss


def test_stability_with_matching_configurations(model, gen_model, input_coefficient, rate_factor, tau_vr, learn_rate, optim):
    t=4000

    gen_model_rate = torch.tensor(rate_factor)
    model_rate = gen_model_rate.clone().detach()
    optims = [optim(list(model.parameters()), lr=learn_rate),
              optim([model_rate], lr=learn_rate)]

    batch_losses = []
    train_iter = 0; avg_batch_loss = 10
    while train_iter < 100 and avg_batch_loss > 5.0:
        gen_inputs = input_coefficient * poisson_input(gen_model_rate, t, gen_model.N)
        gen_model.reset_hidden_state()
        gen_membrane_potentials, targets = generate_model_data(model=gen_model, inputs=gen_inputs)
        # for gen spiketrain this may be thresholded to binary values:
        gen_spiketrain = targets.clone().detach()

        # for neur_ind in range(gen_membrane_potentials.shape[1]):
        #     plot_neuron(gen_membrane_potentials[:, neur_ind].data, title='Generative neuron model #{}'.format(neur_ind),
        #                 fname_ext='_test_2_neuron_{}'.format(neur_ind))
        del gen_membrane_potentials, targets, gen_inputs

        avg_batch_loss = fit_mini_batches(model=model, inputs=None,
                                          target_spiketrain=gen_spiketrain, tau_van_rossum=tau_vr,
                                          current_rate=model_rate, batch_size=500, uuid='test_SNN_fitting_stability',
                                          optimisers=optims)
        model_rate = model_rate.clone().detach()  # reset

        # for param_i, param in enumerate(list(model.parameters())):
        #     # print('parameter #{}: {}'.format(param_i, param))
        #     # if param.grad is not None:
        #     assert param.grad is not None, "gradient was none. param #{}, \nparam: {}\nparam.grad: {}"\
        #         .format(param_i, param, param.grad)
        #     assert torch.abs(param.grad.sum()) != 0, "gradients should not be zero. param #{}, " \
        #                                                 "\nparam: {}\nparam.grad: {}".format(param_i, param, param.grad)

        train_iter += 1
        batch_losses.append(avg_batch_loss)

    plot_losses(batch_losses, [], 'test_SNN_fitting_stability',
                custom_title='Avg. batch losses ({}, lr={})'.format(optim.__name__, learn_rate))


gen_model = BaselineSNN.BaselineSNN(device='cpu', parameters={}, N=12, w_mean=0.8, w_var=0.6)
model = BaselineSNN.BaselineSNN(device='cpu', parameters={}, N=12, w_mean=0.8, w_var=0.6)
tau_vr = torch.tensor(20.0)
learn_rate = 0.1
# test_stability_with_matching_configurations_deprecated(gen_model, model, input_coefficient=1.0, rate_factor=0.7, tau_vr=tau_vr, learn_rate=0.05)
test_stability_with_matching_configurations(gen_model, model, input_coefficient=1.0, rate_factor=0.7, tau_vr=tau_vr,
                                            learn_rate=learn_rate, optim=torch.optim.SGD)

# gen_model = LIF.LIF(device='cpu', parameters={}, tau_m=6.5, N=3, w_mean=0.2, w_var=0.25)
# model = LIF.LIF(device='cpu', parameters={}, tau_m=6.5, N=3, w_mean=0.2, w_var=0.25)
# test_stability_with_matching_configurations_and_training_input_noise(gen_model, model, input_coefficient=4.0, rate_factor=0.5)
# test_stability_with_matching_configurations_different_training_input_noise(gen_model, model, input_coefficient=4.0, rate_factor=0.5)
#
# gen_model = Izhikevich.Izhikevich(device='cpu', parameters={}, N=3, tau_g=1.0, a=0.1, b=0.27, w_mean=0.15, w_var=0.25)
# model = Izhikevich.Izhikevich(device='cpu', parameters={}, N=3, tau_g=1.0, a=0.1, b=0.27, w_mean=0.15, w_var=0.25)
# test_stability_with_matching_configurations_and_training_input_noise(gen_model, model, input_coefficient=1.0, rate_factor=0.25)
# test_stability_with_matching_configurations_different_training_input_noise(gen_model, model, input_coefficient=1.0, rate_factor=0.25)

# gen_model = Izhikevich.Izhikevich_constrained(device='cpu', parameters={}, N=3, a=0.1, b=0.28, w_mean=0.1, w_var=0.25)
# model = Izhikevich.Izhikevich_constrained(device='cpu', parameters={}, N=3, a=0.1, b=0.28, w_mean=0.1, w_var=0.25)
# test_stability_with_matching_configurations_and_training_input_noise(gen_model, model, rate_factor=0.25)
# test_stability_with_matching_configurations_different_training_input_noise(gen_model, model, rate_factor=0.25)
