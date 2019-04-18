from collections import OrderedDict
import numpy as np
import theano
from theano import config
import theano.tensor as T

theano.config.optimizer = 'fast_compile'
#theano.config.exception_verbosity = 'high'
#theano.config.compute_test_value = 'warn'

config.floatX = 'float32'



def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent
    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.
    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')
    return f_grad_shared, f_update

def adadelta(lr, tparams, grads, x, mask, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / T.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update

def glorot_uniform_T(shape):
    fan_in, fan_h, fan_out = shape[0], shape[1], shape[2]
    scale = np.sqrt(6. / (fan_in + fan_h + fan_out))
    return np.random.uniform(low=-scale, high=scale, size=shape).astype(config.floatX)

def glorot_uniform(shape):
    fan_in, fan_out = shape[0], shape[1]
    scale = np.sqrt(6. / (fan_in + fan_out))
    return np.random.uniform(low=-scale, high=scale, size=shape).astype(config.floatX)

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)

def _p(pp, name):
    return '%s_%s' % (pp, name)

def init_params_o2(params, ninp, nhid, prefix):
    W = glorot_uniform_T([ninp, nhid, nhid])
    B = 2*np.ones(nhid, dtype=config.floatX)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'B')] = B
    return params

def init_params_elman(params, ninp, nhid, prefix):
    W_i = glorot_uniform([ninp, nhid])
    W_h = glorot_uniform([nhid, nhid])
    B = 2*np.ones(nhid, dtype=config.floatX)
    params[_p(prefix, 'W_i')] = W_i
    params[_p(prefix, 'W_h')] = W_h
    params[_p(prefix, 'B')] = B
    return params

def init_params_mirnn(params, ninp, nhid, prefix):
    W_i = glorot_uniform([ninp, nhid])
    W_h = glorot_uniform([nhid, nhid])
    B = 2*np.ones(nhid, dtype=config.floatX)
    alpha = 2 * np.ones(nhid, dtype=config.floatX)
    beta1 = 0.5 * np.ones(nhid, dtype=config.floatX)
    beta2 = 0.5 * np.ones(nhid, dtype=config.floatX)

    params[_p(prefix, 'W_i')] = W_i
    params[_p(prefix, 'W_h')] = W_h
    params[_p(prefix, 'B')] = B
    params[_p(prefix, 'alpha')] = alpha
    params[_p(prefix, 'beta1')] = beta1
    params[_p(prefix, 'beta2')] = beta2
    return params

def init_params_lstm(params, ninp, nhid, prefix):
    U_i = glorot_uniform([ninp, nhid])
    U_f = glorot_uniform([ninp, nhid])
    U_o = glorot_uniform([ninp, nhid])
    U_g = glorot_uniform([ninp, nhid])

    W_i = glorot_uniform([nhid, nhid])
    W_f = glorot_uniform([nhid, nhid])
    W_o = glorot_uniform([nhid, nhid])
    W_g = glorot_uniform([nhid, nhid])

    params[_p(prefix, 'U_i')] = U_i
    params[_p(prefix, 'U_f')] = U_f
    params[_p(prefix, 'U_o')] = U_o
    params[_p(prefix, 'U_g')] = U_g

    params[_p(prefix, 'W_i')] = W_i
    params[_p(prefix, 'W_f')] = W_f
    params[_p(prefix, 'W_o')] = W_o
    params[_p(prefix, 'W_g')] = W_g
    return params


def init_params_gru(params, ninp, nhid, prefix):
    U_z = glorot_uniform([ninp, nhid])
    U_r = glorot_uniform([ninp, nhid])
    U_h = glorot_uniform([ninp, nhid])

    W_z = glorot_uniform([nhid, nhid])
    W_r = glorot_uniform([nhid, nhid])
    W_h = glorot_uniform([nhid, nhid])

    params[_p(prefix, 'U_z')] = U_z
    params[_p(prefix, 'U_r')] = U_r
    params[_p(prefix, 'U_h')] = U_h

    params[_p(prefix, 'W_z')] = W_z
    params[_p(prefix, 'W_r')] = W_r
    params[_p(prefix, 'W_h')] = W_h
    return params

def rnn_o2_sig_cell(x_, m_, h_, w, b):
    wh = T.tensordot(x_, w, [[1], [0]])
    h_pre = T.batched_dot(wh, h_) + b
    h = T.nnet.sigmoid(h_pre)
    h = m_[:, None] * h + (1. - m_[:, None]) * h_
    return h

def rnn_o2_tanh_cell(x_, m_, h_, w, b):
    wh = T.tensordot(x_, w, [[1], [0]])
    h_pre = T.batched_dot(wh, h_) + b
    h_rest = T.tanh(h_pre[:, 1:])
    h_0 = T.nnet.sigmoid(h_pre[:, 0])
    h = T.concatenate((h_0[:, None], h_rest), axis=-1)
    h = m_[:, None] * h + (1. - m_[:, None]) * h_
    return h


def rnn_o2_relu_cell(x_, m_, h_, w, b):
    wh = T.tensordot(x_, w, [[1], [0]])
    h_pre = T.batched_dot(wh, h_) + b
    h_rest = T.nnet.relu(h_pre[:, 1:])
    h_0 = T.nnet.sigmoid(h_pre[:, 0])
    h = T.concatenate((h_0[:, None], h_rest), axis=-1)
    h = m_[:, None] * h + (1. - m_[:, None]) * h_
    return h

def rnn_elman_sig_cell(x_, m_, h_, w_i, w_h, b):
    h_pre = T.dot(h_, w_h) + T.dot(x_, w_i) + b
    h = T.nnet.sigmoid(h_pre)
    h = m_[:, None] * h + (1. - m_[:, None]) * h_
    return h

def rnn_elman_tanh_cell(x_, m_, h_, w_i, w_h, b):
    h_pre = T.dot(h_, w_h) + T.dot(x_, w_i) + b
    h_rest = T.tanh(h_pre[:, 1:])
    h_0 = T.nnet.sigmoid(h_pre[:, 0])
    h = T.concatenate((h_0[:, None], h_rest), axis=-1)
    h = m_[:, None] * h + (1. - m_[:, None]) * h_
    return h

def rnn_elman_relu_cell(x_, m_, h_, w_i, w_h, b):
    h_pre = T.dot(h_, w_h) + T.dot(x_, w_i) + b
    h_rest = T.nnet.relu(h_pre[:, 1:])
    h_0 = T.nnet.sigmoid(h_pre[:, 0])
    h = T.concatenate((h_0[:, None], h_rest), axis=-1)
    h = m_[:, None] * h + (1. - m_[:, None]) * h_
    return h

def rnn_mirnn_cell(x_, m_, h_, w_i, w_h, b, alpha_, beta1_, beta2_):
    h_pre = alpha_ * T.dot(x_, w_i) * T.dot(h_, w_h) + \
        beta1_ * T.dot(h_, w_h) + beta2_ * T.dot(x_, w_i) + b
    h_rest = T.tanh(h_pre[:, 1:])
    h_0 = T.nnet.sigmoid(h_pre[:, 0])
    h = T.concatenate((h_0[:, None], h_rest), axis=-1)
    h = m_[:, None] * h + (1. - m_[:, None]) * h_
    return h

def rnn_lstm_cell(x_, m_, h_, c_, u_i, u_f, u_o, u_g, w_i, w_f, w_o, w_g):
    i = T.nnet.sigmoid(T.dot(x_, u_i) + T.dot(h_, w_i))
    f = T.nnet.sigmoid(T.dot(x_, u_f) + T.dot(h_, w_f))
    o = T.nnet.sigmoid(T.dot(x_, u_o) + T.dot(h_, w_o))
    g = T.tanh(T.dot(x_, u_g) + T.dot(h_, w_g))

    c = c_ * f + g * i
    h_rest = T.tanh(c[:, 1:]) * o[:, 1:]
    h_0 = T.tanh(c[:, 0]) * o[:, 0]
    h = T.concatenate((h_0[:, None], h_rest), axis=-1)
    h = m_[:, None] * h + (1. - m_[:, None]) * h_
    return [h, c]

def rnn_gru_cell(x_, m_, h_, u_z, u_r, u_h, w_z, w_r, w_h):
    z = T.nnet.sigmoid(T.dot(x_, u_z) + T.dot(h_, w_z))
    r = T.nnet.sigmoid(T.dot(x_, u_r) + T.dot(h_, w_r))
    c = T.tanh(T.dot(x_, u_h) + T.dot(h_ * r, w_h))

    h_rest = (T.ones_like(z[:, 1:]) - z[:, 1:]) * c[:, 1:] + z[:, 1:] * h_[:, 1:]
    h_0 = (T.ones_like(z[:, 0]) - z[:, 0]) * c[:, 0] + z[:, 0] * h_[:, 0]
    h = T.concatenate((h_0[:, None], h_rest), axis=-1)
    h = m_[:, None] * h + (1. - m_[:, None]) * h_
    return h

class RNNModel():

    def __init__(self, rnn_type, ntoken, ninp, nhid, seed=666, lambda_value=0.5, debug=False):

        np.random.seed(seed)
        self.params = OrderedDict()

        if rnn_type == 'o2_sig':
            self.recurrent_cell = rnn_o2_sig_cell
            self.params = init_params_o2(self.params, ninp, nhid, rnn_type)
        elif rnn_type == 'o2_tanh':
            self.recurrent_cell = rnn_o2_tanh_cell
            self.params = init_params_o2(self.params, ninp, nhid, rnn_type)
        elif rnn_type == 'o2_relu':
            self.recurrent_cell = rnn_o2_relu_cell
            self.params = init_params_o2(self.params, ninp, nhid, rnn_type)
        elif rnn_type == 'elman_sig':
            self.recurrent_cell = rnn_elman_sig_cell
            self.params = init_params_elman(self.params, ninp, nhid, rnn_type)
        elif rnn_type == 'elman_tanh':
            self.recurrent_cell = rnn_elman_tanh_cell
            self.params = init_params_elman(self.params, ninp, nhid, rnn_type)
        elif rnn_type == 'elman_relu':
            self.recurrent_cell = rnn_elman_relu_cell
            self.params = init_params_elman(self.params, ninp, nhid, rnn_type)
        elif rnn_type == 'mirnn':
            self.recurrent_cell = rnn_mirnn_cell
            self.params = init_params_mirnn(self.params, ninp, nhid, rnn_type)
        elif rnn_type == 'lstm':
            self.recurrent_cell = rnn_lstm_cell
            self.params = init_params_lstm(self.params, ninp, nhid, rnn_type)
        elif rnn_type == 'gru':
            self.recurrent_cell = rnn_gru_cell
            self.params = init_params_gru(self.params, ninp, nhid, rnn_type)
        else:
            print('model not available')
            exit(0)

        self.prefix = rnn_type
        self.ntoken = ntoken
        self.ninp = ninp
        self.nhid = nhid

        self.optimizer = rmsprop
        self.lambda_value = lambda_value
        self.debug = debug

        self.init_tparams()

    def init_tparams(self):
        self.tparams = OrderedDict()
        for kk, pp in self.params.items():
            self.tparams[kk] = theano.shared(self.params[kk], name=kk)
            if self.debug:
                self.tparams[kk].tag.test_value = self.params[kk]

    def init_hidden(self, n_samples, h_seed):
        np.random.seed(h_seed)
        h_init_tmp = np.random.uniform(low=1e-5, high=1.0, size=(1, self.nhid-1)).astype(config.floatX)
        h_init = np.hstack((np.ones([1,1]), h_init_tmp)).astype(config.floatX)
        self.h_init = np.tile(h_init, (n_samples, 1))

    def reload_hidden(self, h_init, n_samples):
        self.h_init = np.tile(np.reshape(h_init,(1,-1)), (n_samples, 1))

    def build_model(self):

        lr = T.scalar(name='lr')
        x = T.tensor3('x', dtype=config.floatX)
        mask = T.matrix('mask', dtype=config.floatX)
        y = T.vector('y', dtype='int32')

        if self.debug:
            lr.tag.test_value = np.random.rand(1)
            x.tag.test_value = np.random.randint(low=0, high=2, size=(50, self.h_init.shape[0], self.ninp), dtype='int32')
            x.tag.test_value = x.tag.test_value.astype(config.floatX)
            mask.tag.test_value = np.ones((50, self.h_init.shape[0]),
                                          dtype=config.floatX)
            mask.tag.test_value[:-5, :] = 0.0

            y.tag.test_value = np.random.randint(low=0, high=2,
                                                 size=(self.h_init.shape[0],),
                                                 dtype='int32')

        #n_timesteps = x.shape[0]
        #n_samples = x.shape[1]

        h = self.build_layer(x, mask, prefix=self.prefix)

        #loss = T.sum(((y - h[-1, :, 0]) ** 2) / 2)
        loss = T.sum((y - h[-1, :, 0]) ** 2)
        pred = h[-1, :, 0]
        grads = T.grad(loss, wrt=list(self.tparams.values()))

        self.f_states = theano.function([x, mask], outputs=h, name='f_states')
        self.f_pred = theano.function([x, mask], outputs=pred, name='f_pred')
        self.f_grad = theano.function([x, mask, y], outputs=grads, name='f_grad')
        self.f_grad_shared, self.f_update = \
            self.optimizer(lr, self.tparams, grads, x, mask, y, loss)


    def build_layer(self, x, mask, prefix):
        nsteps = x.shape[0]
        if x.ndim == 3:
            n_samples = x.shape[1]
        else:
            n_samples = 1

        if prefix == 'o2_sig':
            h, updates = theano.scan(self.recurrent_cell, sequences=[x, mask],
                                     non_sequences=[self.tparams[_p(prefix, 'W')],
                                                    self.tparams[_p(prefix, 'B')]],
                                     outputs_info=self.h_init, name=_p(prefix, '_layers'),
                                     n_steps=nsteps)
            return h

        elif prefix == 'o2_tanh':
            h, updates = theano.scan(self.recurrent_cell, sequences=[x, mask],
                                     non_sequences=[self.tparams[_p(prefix, 'W')],
                                                    self.tparams[_p(prefix, 'B')]],
                                     outputs_info=self.h_init, name=_p(prefix, '_layers'),
                                     n_steps=nsteps)
            return h
        elif prefix == 'o2_relu':
            h, updates = theano.scan(self.recurrent_cell, sequences=[x, mask],
                                     non_sequences=[self.tparams[_p(prefix, 'W')],
                                                    self.tparams[_p(prefix, 'B')]],
                                     outputs_info=self.h_init, name=_p(prefix, '_layers'),
                                     n_steps=nsteps)
            return h
        elif prefix == 'elman_sig':
            h, updates = theano.scan(self.recurrent_cell, sequences=[x, mask],
                                     non_sequences=[self.tparams[_p(prefix, 'W_i')],
                                                    self.tparams[_p(prefix, 'W_h')],
                                                    self.tparams[_p(prefix, 'B')]],
                                     outputs_info=self.h_init, name=_p(prefix, '_layers'),
                                     n_steps=nsteps)
            return h
        elif prefix == 'elman_tanh':
            h, updates = theano.scan(self.recurrent_cell, sequences=[x, mask],
                                     non_sequences=[self.tparams[_p(prefix, 'W_i')],
                                                    self.tparams[_p(prefix, 'W_h')],
                                                    self.tparams[_p(prefix, 'B')]],
                                     outputs_info=self.h_init, name=_p(prefix, '_layers'),
                                     n_steps=nsteps)
            return h
        elif prefix == 'elman_relu':
            h, updates = theano.scan(self.recurrent_cell, sequences=[x, mask],
                                     non_sequences=[self.tparams[_p(prefix, 'W_i')],
                                                    self.tparams[_p(prefix, 'W_h')],
                                                    self.tparams[_p(prefix, 'B')]],
                                     outputs_info=self.h_init, name=_p(prefix, '_layers'),
                                     n_steps=nsteps)
            return h
        elif prefix == 'mirnn':
            h, updates = theano.scan(self.recurrent_cell, sequences=[x, mask],
                                     non_sequences=[self.tparams[_p(prefix, 'W_i')],
                                                    self.tparams[_p(prefix, 'W_h')],
                                                    self.tparams[_p(prefix, 'B')],
                                                    self.tparams[_p(prefix, 'alpha')],
                                                    self.tparams[_p(prefix, 'beta1')],
                                                    self.tparams[_p(prefix, 'beta2')]],
                                     outputs_info=self.h_init, name=_p(prefix, '_layers'),
                                     n_steps=nsteps)
            return h
        elif prefix == 'lstm':
            [h, c], updates = theano.scan(self.recurrent_cell, sequences=[x, mask],
                                          non_sequences=[self.tparams[_p(prefix, 'U_i')],
                                                         self.tparams[_p(prefix, 'U_f')],
                                                         self.tparams[_p(prefix, 'U_o')],
                                                         self.tparams[_p(prefix, 'U_g')],
                                                         self.tparams[_p(prefix, 'W_i')],
                                                         self.tparams[_p(prefix, 'W_f')],
                                                         self.tparams[_p(prefix, 'W_o')],
                                                         self.tparams[_p(prefix, 'W_g')]],
                                          outputs_info=[self.h_init, self.h_init],
                                          name=_p(prefix, '_layers'), n_steps=nsteps)
            return T.concatenate((h, c), axis=-1)

        elif prefix == 'gru':
            h, updates = theano.scan(self.recurrent_cell, sequences=[x, mask],
                                     non_sequences=[self.tparams[_p(prefix, 'U_z')],
                                                    self.tparams[_p(prefix, 'U_r')],
                                                    self.tparams[_p(prefix, 'U_h')],
                                                    self.tparams[_p(prefix, 'W_z')],
                                                    self.tparams[_p(prefix, 'W_r')],
                                                    self.tparams[_p(prefix, 'W_h')]],
                                     outputs_info=self.h_init, name=_p(prefix, '_layers'),
                                     n_steps=nsteps)
            return h
        else:
            print('model not available')
            exit(0)
'''
if __name__ == '__main__':

    model = RNNModel(rnn_type = 'o2_sig', ntoken=2, ninp=2, nhid=10,
                     ranSeed = 666, lambda_value = 0.5, debug = True)
    model.init_hidden(10)
    model.build_model()
'''