import torch

from Log import Logger

LOG = Logger('gradient_clamping_errors')


def static_clamp_for(grad, l, m, p):
    clamped_grad = grad.clone().detach()
    # p = pv()
    for i in range(grad.shape[0]):
        if clamped_grad[i] < l-p[i] or clamped_grad[i] > m-p[i]:
            LOG.log('PARAMETER ABOUT TO GO OUT OF BOUNDS. p: {},\nl: {}, m: {},\ngrad:{}'.format(p, l, m, grad))
        clamped_grad[i].clamp_(l - p[i], m - p[i])
        # grad[i].clamp_(l - p[i], m - p[i])
    return clamped_grad
    # return grad
    # return torch.where((grad < -1e-08) + (1e-08 < grad), grad.clamp(float(l.clone()-p.clone()), float(m.clone()-p.clone())), grad.clone())
