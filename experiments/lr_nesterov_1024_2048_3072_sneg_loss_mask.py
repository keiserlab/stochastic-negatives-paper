import sys
import lasagne
from lasagne_nn.hdf5_basic_nn import build_nn as bnn
from lasagne_nn.hdf5_basic_nn import parse_args, train, TRAIN_PERCENTAGE
from common.h5py_data_loader import H5pyDataLoader, SingleValueDistributionFunc
from lasagne_nn.train import (ThresholdStochasticNegativeError, GenericUpdateWithRegularization,
                              masked_loss_with_regularization)

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
STOCHASTIC_NEGATIVE_VALUE = -3.0
STOCHASTIC_NEGATIVE_THRESHOLD = 5.0
STOCHASTIC_NEGATIVE_CENTER = 3.0
SQUARED_ERROR_OBJECTIVE = lasagne.objectives.squared_error

def build_data_loader(training_data,
                      test_indices_filename=None,
                      npKi=False,
                      multitask=False,
                      stochastic_negatives=False,
                      target_map_file=None,
                      train_percentage=TRAIN_PERCENTAGE,
                      negative_blacklist_file=None,
                      negative_threshold=None,
                      positive_negative_ratio=None,
                      random_seed=None,
                      stochastic_negative_value=STOCHASTIC_NEGATIVE_VALUE,
                      **kwargs):
    single_value_dist = SingleValueDistributionFunc(value=stochastic_negative_value)

    data_loader = H5pyDataLoader(
        hdf5_file=training_data,
        target_map_file=target_map_file,
        train_percentage=train_percentage,
        test_indices_file=test_indices_filename,
        random_seed=random_seed,
        multitask=multitask,
        npKi=npKi,
        stochastic_negatives=stochastic_negatives,
        negative_blacklist_file=negative_blacklist_file,
        negative_threshold=negative_threshold,
        positive_negative_ratio=positive_negative_ratio,
        stochastic_negative_distribution=single_value_dist
    )
    return data_loader


def build_nn(input_shape, output_shape, input_var=None):
    return bnn(input_shape, output_shape, input_var=input_var,  use_dropout=USE_DROPOUT,
               hidden_layers=HIDDEN_LAYERS, output_activation=OUTPUT_ACTIVATION)


def build_sq_err_nesterov_with_sneg_mask(
        learning_rate, momentum, regularization=None,
        objective=SQUARED_ERROR_OBJECTIVE,
        stochastic_negative_value=STOCHASTIC_NEGATIVE_VALUE,
        stochastic_negative_threshold=STOCHASTIC_NEGATIVE_THRESHOLD,
        stochastic_negative_center=STOCHASTIC_NEGATIVE_CENTER):
    masked_sneg_squared_err = ThresholdStochasticNegativeError(
        objective=objective,
        stochastic_negative_value=stochastic_negative_value,
        stochastic_negative_threshold=stochastic_negative_threshold,
        stochastic_negative_center=stochastic_negative_center
    )
    return GenericUpdateWithRegularization(
        loss_function=masked_loss_with_regularization,
        objective=masked_sneg_squared_err,
        update=dict(
            function=lasagne.updates.nesterov_momentum,
            kwargs={'learning_rate': learning_rate, 'momentum': momentum}),
        regularization=regularization)


def main():
    args, kwargs = parse_args(learning_rate=LEARNING_RATE, momentum=MOMENTUM,
                              regularization=REGULARIZATION)
    kwargs['build_nn_func'] = build_nn
    learning_rate = kwargs['build_train_test_func'].update['kwargs']['learning_rate']
    momentum = kwargs['build_train_test_func'].update['kwargs']['momentum']
    kwargs['build_train_test_func'] = build_sq_err_nesterov_with_sneg_mask(
        learning_rate, momentum)
    kwargs['data_loader'] = build_data_loader(args[0], **kwargs)
    train(*args, **kwargs)


if __name__ == "__main__":
    sys.exit(main())
