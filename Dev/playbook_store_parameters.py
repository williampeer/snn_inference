import torch
import torch.nn as nn
from torch import tensor as T

from Models.GLIF import GLIF

m = GLIF(device='cpu', parameters={})
t = m.C_m.clone().detach()
sut = m.state_dict()
print(sut['C_m'])
m.C_m = nn.Parameter(torch.rand((m.N,)), requires_grad=True)
t2 = m.C_m.clone().detach()
assert torch.mean(t) != torch.mean(t2)
print(m.state_dict()['C_m'])
m.load_state_dict(sut)
t3 = m.C_m.clone().detach()
assert torch.mean(t) == torch.mean(t3)
print(m.state_dict()['C_m'])
