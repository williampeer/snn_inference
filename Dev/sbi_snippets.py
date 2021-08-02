import torch
import numpy as np

from sbi import utils
from sbi import analysis
from sbi import inference
from sbi.inference import SNPE, simulate_for_sbi, prepare_for_sbi
from sbi import utils as utils
from sbi.analysis import pairplot, conditional_pairplot, conditional_corrcoeff

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, rc
# -----------------------------------------
from sbi.utils.get_nn_models import posterior_nn  # For SNLE: likelihood_nn(). For SNRE: classifier_nn()

density_estimator_build_fun = posterior_nn(model='nsf', hidden_features=60, num_transforms=3)
inference = SNPE(prior=prior, density_estimator=density_estimator_build_fun)

# -----------------------------------------
# set seed for numpy and torch
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
# -----------------------------------------
# ===== Note: If we want to define an embedding net over summary statistics =====
# # set prior distribution for the parameters
# prior = utils.BoxUniform(low=torch.tensor([0.0, 0.0]),
#                              high=torch.tensor([1.0, 2*np.pi]))
#
# # make a SBI-wrapper on the simulator object for compatibility
# simulator_wrapper, prior = prepare_for_sbi(simulator_model, prior)
#
# # instantiate the neural density estimator
# neural_posterior = utils.posterior_nn(model='maf',
#                                       embedding_net=embedding_net,
#                                       hidden_features=10,
#                                       num_transforms=2)
#
# # setup the inference procedure with the SNPE-C procedure
# inference = SNPE(prior=prior, density_estimator=neural_posterior)
#
# # run the inference procedure on one round and 10000 simulated data points
# theta, x = simulate_for_sbi(simulator_wrapper, prior, num_simulations=10000)
# density_estimator = inference.append_simulations(theta, x).train()
# posterior = inference.build_posterior(density_estimator)
# -----------------------------------------
# -----------------------------------------
# ==== Cool animation snippet
rc('animation', html='html5')

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim((-2, 2))
ax.set_ylim((-2, 2))

def init():
    line, = ax.plot([], [], lw=2)
    line.set_data([], [])
    return (line,)

def animate(angle):
    num_samples_vis = 1000
    line = ax.scatter(posterior_samples[:num_samples_vis, 0], posterior_samples[:num_samples_vis, 1], posterior_samples[:num_samples_vis, 2], zdir='z', s=15, c='#2171b5', depthshade=False)
    ax.view_init(20, angle)
    return (line,)

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=range(0,360,5), interval=150, blit=True)

plt.close()
# -----------------------------------------