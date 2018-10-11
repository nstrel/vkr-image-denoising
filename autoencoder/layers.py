import numpy as np
import theano as th
import theano.tensor as T


class ConvLayer3D(object):
    def output_shp(self):
        i_shp = np.asarray(self.image_shp[2:])
        f_shp = np.asarray(self.filter_shp[2:])
        s_shp = np.asarray(self.subsample)
        if self.border_mode == 'valid':
            self.border_mode = np.zeros_like(i_shp)
        elif self.border_mode == 'full':
            self.border_mode = f_shp - 1
        elif self.border_mode == 'half':
            self.border_mode = f_shp // 2

        b_shp = np.asarray(self.border_mode)
        # O = (I - F + 2B)//S + 1
        o_shp = (i_shp - f_shp + 2 * b_shp) // s_shp + 1

        o_shp = tuple(map(int, o_shp))
        res = (self.image_shp[0], self.filter_shp[0], *o_shp)
        return res

    def __init__(self, rng, input, filter_shape, image_shape, **conv_params):
        self.input = input
        self.filter_shp = filter_shape
        self.image_shp = image_shape
        self.border_mode = conv_params.get('border_mode', 'valid')
        self.subsample = conv_params.get('subsample', (1, 1, 1))
        W_bound = np.sqrt(
            6. / (filter_shape[0] * np.prod(filter_shape[2:])
                  + np.prod(filter_shape[1:])))
        self.W = th.shared(
            np.asarray(
                rng.uniform(
                    low=-W_bound,
                    high=W_bound,
                    size=filter_shape
                ),
                dtype=th.config.floatX
            ),
            borrow=True
        )
        self.b = th.shared(
            value=np.zeros(
                shape=(filter_shape[0]),
                dtype=th.config.floatX
            )
        )
        conv_out = T.nnet.conv3d(
            self.input, self.W, self.image_shp, self.filter_shp, **conv_params
        )
        # (mini-batch size, channels, depth, height, width)
        self.output = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x', 'x'))
        self.params = [self.W, self.b]
        self.func = th.function(
            [self.input],
            self.output
        )


class DeconvLayer3D(object):
    def __init__(self, rng, input, filter_shape, image_shape, **conv_params):
        self.input = input
        self.filter_shp = filter_shape
        self.image_shp = image_shape
        W_bound = np.sqrt(
            6. / (filter_shape[0] * np.prod(filter_shape[2:])
                  + np.prod(filter_shape[1:])))
        self.W = th.shared(
            np.asarray(
                rng.uniform(
                    low=-W_bound,
                    high=W_bound,
                    size=filter_shape
                ),
                dtype=th.config.floatX
            ),
            borrow=True
        )
        self.b = th.shared(
            value=np.zeros(
                shape=(image_shape[2],),
                dtype=th.config.floatX
            )
        )

        conv_in = T.nnet.abstract_conv.conv3d_grad_wrt_inputs(
            self.input, self.W, self.image_shp, self.filter_shp, **conv_params
        )

        self.output = T.tanh(conv_in + self.b.dimshuffle('x', 'x', 0, 'x', 'x'))
        self.params = [self.W, self.b]
        self.func = th.function(
            [self.input],
            self.output
        )
