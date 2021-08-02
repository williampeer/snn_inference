import torch
import numpy as np


# -----------------------------------------
# create the figure
fig, ax = analysis.pairplot(samples,
                            points=true_parameter,
                            labels=['r', r'$\theta$'],
                            limits=[[0, 1], [0, 2*np.pi]],
                            points_colors='r',
                            points_offdiag={'markersize': 6},
                            figsize=[7.5, 6.4])

# -----------------------------------------
posterior_samples = posterior.sample((5000,))

fig, ax = pairplot(
    samples=posterior_samples,
    limits=torch.tensor([[-2., 2.]]*3),
    upper=['kde'],
    diag=['kde'],
    figsize=(5,5)
)
# -----------------------------------------
corr_matrix_marginal = np.corrcoef(posterior_samples.T)
fig, ax = plt.subplots(1,1, figsize=(4, 4))
im = plt.imshow(corr_matrix_marginal, clim=[-1, 1], cmap='PiYG')
_ = fig.colorbar(im)
# -----------------------------------------
condition = posterior.sample((1,))

_ = conditional_pairplot(
    density=posterior,
    condition=condition,
    limits=torch.tensor([[-2., 2.]]*3),
    figsize=(5,5)
)
# -----------------------------------------
cond_coeff_mat = conditional_corrcoeff(
    density=posterior,
    condition=condition,
    limits=torch.tensor([[-2., 2.]]*3),
)
fig, ax = plt.subplots(1,1, figsize=(4,4))
im = plt.imshow(cond_coeff_mat, clim=[-1, 1], cmap='PiYG')
_ = fig.colorbar(im)
# -----------------------------------------
