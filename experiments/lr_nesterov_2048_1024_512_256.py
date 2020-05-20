import sys
import lasagne
from lasagne_nn.hdf5_basic_nn import build_nn as bnn
from lasagne_nn.hdf5_basic_nn import parse_args, train

TRAIN_PERCENTAGE = .8
BATCH_SIZE = 300  # larger=faster epochs; smaller=better loss/epoch
NUM_EPOCHS = 300
LEARNING_RATE = 0.1
MOMENTUM = 0.5
REGULARIZATION = {'lambda': 0.000001, 'penalty': lasagne.regularization.l2}
USE_DROPOUT = True
OUTPUT_ACTIVATION = lasagne.nonlinearities.leaky_rectify
HIDDEN_LAYERS = [
    # (num_units, dropout_p, nonlinearity, regularization)
    (2048, .10, lasagne.nonlinearities.leaky_rectify),
    (1024, .25, lasagne.nonlinearities.leaky_rectify),
    (512, .25, lasagne.nonlinearities.leaky_rectify),
    (256, .25, lasagne.nonlinearities.leaky_rectify)
]


def build_nn(input_shape, output_shape, input_var=None):
    return bnn(input_shape, output_shape, input_var=input_var,  use_dropout=USE_DROPOUT,
               hidden_layers=HIDDEN_LAYERS, output_activation=OUTPUT_ACTIVATION)


def main(argv=sys.argv[1:]):
    args, kwargs = parse_args(argv)
    kwargs['build_nn_func'] = build_nn
    train(*args, **kwargs)


if __name__ == "__main__":
    sys.exit(main())
