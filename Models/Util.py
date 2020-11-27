import torch


def static_clamp_for(grad, l, m, p):
    # clamped_grad = grad.clone().detach()
    # for i in range(p.shape[0]):
    #     clamped_grad.clamp_(l - p[i], m - p[i])
    clamped_grad = torch.where(-1e-08 < grad < 1e-08, grad, grad.clamp(l-p, m-p))

    return clamped_grad