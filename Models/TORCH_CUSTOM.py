from Log import Logger

LOG = Logger('gradient_clamping_errors')


def static_clamp_for(new_grad, l, m, p):
    # return torch.where((grad < -1e-08) + (1e-08 < grad), grad.clamp(float(l.clone()-p.clone()), float(m.clone()-p.clone())), grad.clone())
    for i in range(p.shape[0]):
        out_of_bounds = new_grad[i] < l-p[i] or new_grad[i] > m-p[i]
        if out_of_bounds:
            LOG.log('PARAMETER ABOUT TO GO OUT OF BOUNDS. \np[{}]: {},\nl: {}, m: {},\ngrad[{}]: {}'.format(i, p[i], l, m, i, new_grad[i]))
        new_grad[i].data.clamp_(p[i] - m, p[i] - l)  # apparently, the gradient is subtracted. test with Adam too.
        # new_grad[i].data.clamp_(l - p[i], m - p[i])
        if out_of_bounds:
            LOG.log('clamped grad {}: {}'.format(i, new_grad[i].data))

    return new_grad.data


def static_clamp_for_vector_bounds(new_grad, l, m, p):
    # return torch.where((grad < -1e-08) + (1e-08 < grad), grad.clamp(float(l.clone()-p.clone()), float(m.clone()-p.clone())), grad.clone())
    for i in range(p.shape[0]):
        out_of_bounds = new_grad[i] < l[i]-p[i] or new_grad[i] > m[i]-p[i]
        if out_of_bounds:
            LOG.log('PARAMETER ABOUT TO GO OUT OF BOUNDS. \ni: {}, p[i]: {},\nl: {}, m: {},\ngrad[i]: {}'.format(i, p[i], l[i], m[i], new_grad[i]))
        new_grad[i].data.clamp_(p[i] - m[i], p[i] - l[i])  # apparently, the gradient is subtracted. test with Adam too.
        # new_grad[i].data.clamp_(l - p[i], m - p[i])
        if out_of_bounds:
            LOG.log('clamped grad {}: {}'.format(i, new_grad[i].data))

    return new_grad.data
