import IO
from Models import SleepModels

LIF_model = SleepModels.LIF()
IO.save(model=LIF_model, loss=None, uuid='LIF_sleep_model', fname='LIF_sleep_model')

Izhikevich_model = SleepModels.IzhikevichStable()
IO.save(model=Izhikevich_model, loss=None, uuid='Izhikevich_sleep_model', fname='Izhikevich_sleep_model')
