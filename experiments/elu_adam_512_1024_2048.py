import sys
import lasagne
from lasagne_nn.hdf5_basic_nn import build_nn as bnn
from lasagne_nn.hdf5_basic_nn import parse_args, train
from lasagne_nn.train import squared_error_adam_train_test_func


# note: this should probably be run with a learning rate of 0.01 and momentum less than 0.5 to prevent overfitting
USE_DROPOUT = True
OUTPUT_ACTIVATION = lasagne.nonlinearities.elu
HIDDEN_LAYERS = [
    # (num_units, dropout_p, nonlinearity, regularization)
    (512, .10, lasagne.nonlinearities.elu),
    (1024, .25, lasagne.nonlinearities.elu),
    (2048, .25, lasagne.nonlinearities.elu)
]
LEARNING_RATE = 0.001
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1E-8
REGULARIZATION = {'lambda': 0.000001, 'penalty': lasagne.regularization.l2}


def build_nn(input_shape, output_shape, input_var=None):
    return bnn(input_shape, output_shape, input_var=input_var, use_dropout=USE_DROPOUT,
               hidden_layers=HIDDEN_LAYERS, output_activation=OUTPUT_ACTIVATION)


def main(argv=sys.argv[1:]):
    args, kwargs = parse_args(argv)
    kwargs['build_nn_func'] = build_nn
    kwargs['build_train_test_func'] = squared_error_adam_train_test_func(
        learning_rate=LEARNING_RATE,
        beta1=BETA1,
        beta2=BETA2,
        epsilon=EPSILON,
        regularization=REGULARIZATION
    )
    train(*args, **kwargs)


if __name__ == "__main__":
    sys.exit(main())
