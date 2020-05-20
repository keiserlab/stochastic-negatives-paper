import sys
import lasagne
from lasagne_nn.hdf5_basic_nn import build_nn as bnn
from lasagne_nn.hdf5_basic_nn import parse_args, train


# note: this should probably be run with a learning rate of 0.01 and momentum less than 0.5 to prevent overfitting
USE_DROPOUT = True
OUTPUT_ACTIVATION = lasagne.nonlinearities.leaky_rectify
HIDDEN_LAYERS = [
    # (num_units, dropout_p, nonlinearity, regularization)
    (4096, .10, lasagne.nonlinearities.leaky_rectify),
    (3072, .25, lasagne.nonlinearities.leaky_rectify),
    (2048, .25, lasagne.nonlinearities.leaky_rectify),
    (1024, .25, lasagne.nonlinearities.leaky_rectify),
    (512, .25, lasagne.nonlinearities.leaky_rectify)
]


def build_nn(input_shape, output_shape, input_var=None):
    return bnn(input_shape, output_shape, input_var=input_var, use_dropout=USE_DROPOUT,
               hidden_layers=HIDDEN_LAYERS, output_activation=OUTPUT_ACTIVATION)


def main(argv=sys.argv[1:]):
    args, kwargs = parse_args(argv)
    kwargs['build_nn_func'] = build_nn
    train(*args, **kwargs)


if __name__ == "__main__":
    sys.exit(main())
