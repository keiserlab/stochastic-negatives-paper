__author__ = 'Nick'
import theano.tensor as T
import lasagne


class MaskedLayer(lasagne.layers.DenseLayer):
    def __init__(self, incoming, num_units, **kwargs):
        super(MaskedLayer, self).__init__(incoming, num_units, **kwargs)
        self.params[self.W] = self.params[self.W] - {'trainable'}
        self.W_mask = None
        if self.b is not None:
            self.params[self.b] = self.params[self.b] - {'trainable'}
            self.b_mask = None

    def get_params(self, **tags):
        p = super(MaskedLayer, self).get_params(**tags)
        if 'trainable' in tags and tags['trainable']:
            p += [self.W_mask]
            if self.b is not None:
                p += [self.b_mask]
        return p

    def get_output_shape_for(self, input_shape, target_mask=None, **kwargs):
        if target_mask is None:
            return (input_shape[0], self.num_units)
        else:
            return (input_shape[0], target_mask.sum())

    def get_output_for(self, input, target_mask=None, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        if target_mask is None:
            activation = T.dot(input, self.W)
        else:
            self.W_mask = self.W[:, target_mask]
            activation = T.dot(input, self.W_mask)
        if self.b is not None:
            if target_mask is None:
                activation = activation + self.b.dimshuffle('x', 0)
            else:
                self.b_mask = self.b[target_mask]
                activation = activation + self.b_mask.dimshuffle('x', 0)
        return self.nonlinearity(activation)

