import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import glm

import model_util
from TargetModels import TargetEnsembleModels
from experiments import poisson_input

random_seed = 0
snn = TargetEnsembleModels.lif_ensembles_model_dales_compliant(random_seed=random_seed, N=12)
# ext_name = 'ensembles_{}_dales_LIF'.format(random_seed)

rate = 10.
inputs = poisson_input(rate, t=2000, N=snn.N)[:,0]  # rate in Hz
spikes = model_util.feed_inputs_sequentially_return_spiketrain(snn, inputs)[:,5].detach().numpy()
spikes = 1.0 * (spikes > 0.5)
spike_times = np.arange(0, spikes.shape[0], 1) * spikes
print(spikes.sum(), spike_times.sum())
assert spikes.sum() < spike_times.sum()

random_input = poisson_input(rate, t=2000, N=snn.N)[:,0].numpy()
# exog = sm.add_constant(random_input)

formula = 'y ~ x1 + x2'
family = sm.families.Poisson
data = np.stack((random_input, spikes))

glm_model = glm(formula, data, family)
res = glm_model.fit()

sut = glm_model.predict(params=res.params, exog=sm.add_constant(random_input),
                  linear=False)


