import sys
import theano.tensor as T
import lasagne
from lasagne_nn.hdf5_basic_nn import build_nn as bnn
from lasagne_nn.hdf5_basic_nn import parse_args, train
from lasagne_nn.train import GenericUpdateWithRegularization

LEARNING_RATE = 0.05
MOMENTUM = 0.4
REGULARIZATION = {'lambda': 0.0001, 'penalty': lasagne.regularization.l2}
USE_DROPOUT = True
OUTPUT_ACTIVATION = lasagne.nonlinearities.sigmoid
HIDDEN_LAYERS = [
    # (num_units, dropout_p, nonlinearity, regularization)
    (1024, .10, lasagne.nonlinearities.leaky_rectify),
    (2048, .25, lasagne.nonlinearities.leaky_rectify),
    (3072, .25, lasagne.nonlinearities.leaky_rectify)
]
USE_BATCH_NORM = False
BATCH_NORM_ARGS = None
TRAIN_PERCENTAGE = .8
BINARY_THRESHOLD = 5.0


def build_nn(input_shape, output_shape, input_var=None):
    return bnn(input_shape, output_shape,
               input_var=input_var,
               use_dropout=USE_DROPOUT,
               use_batch_norm=USE_BATCH_NORM,
               batch_norm_kwargs=BATCH_NORM_ARGS,
               hidden_layers=HIDDEN_LAYERS,
               output_activation=OUTPUT_ACTIVATION)


class BinaryClassificationTransformer(object):
    def __init__(self, threshold, args_index=1):
        self.threshold = threshold
        self.threshold_t = lasagne.utils.as_theano_expression(lasagne.utils.floatX(self.threshold))
        self.args_index = args_index

    def minibatch_transform(self, *args):
        """Drug Matrix Transformation"""
        return tuple(self.cpu_transform(a) if i == self.args_index else a
                     for i, a in enumerate(args))

    def cpu_transform(self, y):
        return y > self.threshold

    def theano_transform(self, y):
        """theano graph transformation"""
        return T.gt(y, self.threshold_t)

    def objective_transform(self, objective):
        """transform target values to binary before calculating loss"""
        def binary_classification_error(x, y):
            return objective(x, self.theano_transform(y))
        # update name for logging
        binary_classification_error.__name__ = '_'.join(
            [objective.__name__, self.__class__.__name__,
             "threshold_{}".format(self.threshold)])
        return binary_classification_error


def build_train_test(
        learning_rate, momentum, regularization,
        objective=lasagne.objectives.binary_crossentropy,
        binary_threshold=BINARY_THRESHOLD):

    # transform target values to binary before calculating loss
    transformer = BinaryClassificationTransformer(binary_threshold)

    # Masking loss of missing data is handled in GenericUpdateWithRegularization
    return GenericUpdateWithRegularization(
        objective=transformer.objective_transform(objective),
        update=dict(
            function=lasagne.updates.nesterov_momentum,
            kwargs={'learning_rate': learning_rate, 'momentum': momentum}),
        regularization=regularization)


def main():
    args, kwargs = parse_args(learning_rate=LEARNING_RATE, momentum=MOMENTUM,
                              regularization=REGULARIZATION)
    kwargs['build_nn_func'] = build_nn
    regularization = kwargs['build_train_test_func'].regularization
    build_nn_kwargs = kwargs['build_train_test_func'].update['kwargs']
    learning_rate = build_nn_kwargs['learning_rate']
    momentum = build_nn_kwargs['momentum']
    kwargs['build_train_test_func'] = build_train_test(
        learning_rate, momentum, regularization)
    train(*args, **kwargs)


if __name__ == "__main__":
    sys.exit(main())
