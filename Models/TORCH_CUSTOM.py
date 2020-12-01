import torch


def static_clamp_for(grad, l, m, pv):
    # clamped_grad = grad.clone().detach()
    p = pv()
    for i in range(grad.shape[0]):
        # clamped_grad[i].clamp_(l - p[i], m - p[i])
        grad[i] = grad[i].clamp(l - p[i], m - p[i])
    # return clamped_grad
    return grad
    # return torch.where((grad < -1e-08) + (1e-08 < grad), grad.clamp(float(l.clone()-p.clone()), float(m.clone()-p.clone())), grad.clone())