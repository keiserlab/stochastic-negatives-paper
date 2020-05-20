import sys
import logging
import pprint
import theano
import theano.tensor as T
import lasagne
from lasagne_nn.hdf5_basic_nn import build_nn as bnn
from lasagne_nn.hdf5_basic_nn import parse_args, train
from lasagne_nn.train import (GenericUpdateWithRegularization,
                              generic_loss_with_regularization)

LEARNING_RATE = 0.01
MOMENTUM = 0.4
REGULARIZATION = {'lambda': 0.00005, 'penalty': lasagne.regularization.l2}
USE_DROPOUT = True
OUTPUT_ACTIVATION = lasagne.nonlinearities.leaky_rectify
HIDDEN_LAYERS = [
    # (num_units, dropout_p, nonlinearity, regularization)
    (1024, .10, lasagne.nonlinearities.leaky_rectify),
    (2048, .25, lasagne.nonlinearities.leaky_rectify),
    (3072, .25, lasagne.nonlinearities.leaky_rectify)
]

NEGATIVE_THRESHOLD = 5.0
SQUARED_ERROR_OBJECTIVE = lasagne.objectives.squared_error


class IgnoreKnownNegatives(object):
    """
    Weight loss of negative predictions to balance ratio of known positive and negative values in batch.
    """
    def __init__(self, objective, negative_threshold):
        self.objective = objective
        self.negative_threshold = lasagne.utils.as_theano_expression(lasagne.utils.floatX(negative_threshold))
        logging.info(pprint.pformat({'IgnoreKnownNegatives': self.__dict__}))

    def __call__(self, target_prediction, target_known):
        known_positives = T.gt(target_known, self.negative_threshold)
        loss = self.objective(target_prediction, target_known)
        return loss * known_positives


def build_nn(input_shape, output_shape, input_var=None):
    return bnn(input_shape, output_shape, input_var=input_var,  use_dropout=USE_DROPOUT,
               hidden_layers=HIDDEN_LAYERS, output_activation=OUTPUT_ACTIVATION)


def build_sq_err_nesterov_with_weighted_negatives(
        learning_rate, momentum, regularization,
        objective=SQUARED_ERROR_OBJECTIVE,
        negative_threshold=NEGATIVE_THRESHOLD):
    weighted_negatives_squared_err = IgnoreKnownNegatives(
        objective=objective,
        negative_threshold=negative_threshold)
    return GenericUpdateWithRegularization(
        loss_function=generic_loss_with_regularization,
        objective=weighted_negatives_squared_err,
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
    kwargs['build_train_test_func'] = build_sq_err_nesterov_with_weighted_negatives(
        learning_rate, momentum, regularization)
    train(*args, **kwargs)


if __name__ == "__main__":
    sys.exit(main())
