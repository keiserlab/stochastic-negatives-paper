import sys
import lasagne
from lasagne_nn.hdf5_basic_nn import build_nn as bnn
from lasagne_nn.hdf5_basic_nn import parse_args, train

LEARNING_RATE = 0.01
MOMENTUM = 0.4
REGULARIZATION = {'lambda': 0.00005, 'penalty': lasagne.regularization.l2}
USE_DROPOUT = True
OUTPUT_ACTIVATION = lasagne.nonlinearities.leaky_rectify
HIDDEN_LAYERS = [
    # (num_units, dropout_p, nonlinearity, regularization)
    (1024, .10, lasagne.nonlinearities.leaky_rectify),
    (1024, .25, lasagne.nonlinearities.leaky_rectify),
    (2048, .25, lasagne.nonlinearities.leaky_rectify),
    (2048, .25, lasagne.nonlinearities.leaky_rectify),
    (3072, .25, lasagne.nonlinearities.leaky_rectify),
    (3072, .25, lasagne.nonlinearities.leaky_rectify)
]
USE_BATCH_NORM = True
BATCH_NORM_ARGS = None


def build_nn(input_shape, output_shape, input_var=None):
    return bnn(input_shape, output_shape,
               input_var=input_var,
               use_dropout=USE_DROPOUT,
               use_batch_norm=USE_BATCH_NORM,
               batch_norm_kwargs=BATCH_NORM_ARGS,
               hidden_layers=HIDDEN_LAYERS,
               output_activation=OUTPUT_ACTIVATION)


def main():
    args, kwargs = parse_args(learning_rate=LEARNING_RATE, momentum=MOMENTUM,
                              regularization=REGULARIZATION)
    kwargs['build_nn_func'] = build_nn
    train(*args, **kwargs)



if __name__ == "__main__":
    sys.exit(main())
