from fit_to_data_exp_suite import convergence_check

# model = GLIF(device='cpu', parameters={})
# t = 400
# targets = poisson_input(rate=0.6, t=t, N=model.N)
#
# current_rate = torch.tensor(0.5, requires_grad=True)
# optim_params = list(model.parameters())
# # optim_params.append(current_rate)  # TODO: Fix rate optimisation
# optim = torch.optim.Adam(optim_params, lr=0.01)
# constants_van_rossum = Constants(learn_rate=0.01, train_iters=100, N_exp=20, batch_size=200, tau_van_rossum=3.0,
#                                  initial_poisson_rate=0.5, rows_per_train_iter=400, optimiser='Adam',
#                                  loss_fn='van_rossum_dist', evaluate_step=1)
# avg_train_loss = fit_mini_batches(model, inputs=None, target_spiketrain=targets, current_rate=current_rate,
#                                   optimiser=optim, constants=constants_van_rossum, train_i=0, logger=TestLogger())


validation_losses = []
validation_losses.append(3.0)
converged = convergence_check(validation_losses)
assert converged == False, "should not have converged for one validation loss"


validation_losses.append(2.0)
converged = convergence_check(validation_losses)
assert converged == False, "should not have converged for falling validation set loss"

validation_losses.append(2.1)
converged = convergence_check(validation_losses)
assert converged == True, "should have converged for a loss the same or worse for the validation set (suggesting overfitting)"
