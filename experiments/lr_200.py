import sys
import lasagne
import theano.tensor as T
from lasagne_nn.hdf5_basic_nn import build_nn as bnn
from lasagne_nn.hdf5_basic_nn import parse_args, train
from lasagne_nn.utils import EarlyStopping, StoreNetworkParams, StoreLoss, LogEpoch


def elu(x):
    return T.switch(x > 0, x, T.exp(x) - 1)

USE_DROPOUT = False
OUTPUT_ACTIVATION = elu
HIDDEN_LAYERS = [
    # (num_units, dropout_p, nonlinearity, regularization)
    (200, .10, elu),
]
ON_EPOCH_FINISHED = [StoreLoss(every_n_epochs=10),
                     StoreNetworkParams(every_n_epochs=25), LogEpoch(),
                     EarlyStopping(patience=200)]


def build_nn(input_shape, output_shape, input_var=None):
    return bnn(input_shape, output_shape, input_var=input_var, use_dropout=USE_DROPOUT,
               hidden_layers=HIDDEN_LAYERS, output_activation=OUTPUT_ACTIVATION)


def main(argv=sys.argv[1:]):
    args, kwargs = parse_args(argv)
    kwargs['build_nn_func'] = build_nn
    kwargs['on_epoch_finished'] = ON_EPOCH_FINISHED
    train(*args, **kwargs)


if __name__ == "__main__":
    sys.exit(main())
