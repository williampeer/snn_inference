import torch


def static_clamp_for(grad, l, m, p):
    clamped_grad = grad.clone().detach()
    for i in range(grad.shape[0]):
        clamped_grad[i].clamp_(l - p[i], m - p[i])
    return clamped_grad
    # return torch.where((grad < -1e-08) + (1e-08 < grad), grad.clamp(float(l.clone()-p.clone()), float(m.clone()-p.clone())), grad.clone())