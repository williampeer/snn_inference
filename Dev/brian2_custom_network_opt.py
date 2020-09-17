from brian2 import *

# import Dev.brian_GLIF_model_setup
from Dev.brian_GLIF_model_setup import *


def run_simulation(parameters, t_interval=4000*ms):
    # restore()

    # set params?

    S.I_syn = 100 * mA
    run(t_interval)

    spikes = spikemon.spike_trains()
    print('spikemon spike trains:', spikes)
    # return loss_fn()


run_simulation(parameters={})
