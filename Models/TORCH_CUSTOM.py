from Log import Logger

LOG = Logger('gradient_clamping_errors')


def static_clamp_for(new_grad, l, m, p):
    for i in range(p.shape[0]):
        p_out_of_bounds = p[i] < l or m < p[i]
        if p_out_of_bounds:
            LOG.log('PARAMETER OUT OF BOUNDS. \ni: {}, p: {},\nl: {}, m: {},\ngrad: {}'.format(i, p[i], l, m, new_grad[i]))
        new_grad[i].data.clamp_(p[i] - m, p[i] - l)  # apparently, the gradient is subtracted. test with Adam too.

    return new_grad.data


def static_clamp_for_vector_bounds(new_grad, l, m, p):
    for i in range(p.shape[0]):
        p_out_of_bounds = p[i] < l[i] or m[i] < p[i]
        if p_out_of_bounds:
            LOG.log('PARAMETER OUT OF BOUNDS. \ni: {}, p: {},\nl: {}, m: {},\ngrad: {}'.format(i, p[i], l[i], m[i], new_grad[i]))
        new_grad[i].data.clamp_(p[i] - m[i], p[i] - l[i])  # apparently, the gradient is subtracted. test with Adam too.

    return new_grad.data
