import sys
import lasagne
from lasagne_nn.hdf5_basic_nn import build_nn as bnn
from lasagne_nn.hdf5_basic_nn import parse_args, train
from lasagne_nn.train import squared_error_adadelta_train_test_func

# note: this should probably be run with a learning rate of 0.01 and momentum less than 0.5 to prevent overfitting
# even ada delta we want something low!!!
USE_DROPOUT = True
OUTPUT_ACTIVATION = lasagne.nonlinearities.elu
HIDDEN_LAYERS = [
    # (num_units, dropout_p, nonlinearity, regularization)
    (1024, .10, lasagne.nonlinearities.elu),
    (2048, .25, lasagne.nonlinearities.elu),
    (3072, .25, lasagne.nonlinearities.elu)
]

learning_rate = 0.01
rho=0.95
epsilon = 1e-06
REGULARIZATION = {'lambda': 5e-5, 'penalty': lasagne.regularization.l2}


def build_nn(input_shape, output_shape, input_var=None):
    return bnn(input_shape, output_shape, input_var=input_var, use_dropout=USE_DROPOUT,
               hidden_layers=HIDDEN_LAYERS, output_activation=OUTPUT_ACTIVATION)


def main():
    args, kwargs = parse_args()
    kwargs['build_nn_func'] = build_nn
    kwargs['build_train_test_func'] = squared_error_adadelta_train_test_func(
       learning_rate = learning_rate, rho = rho, epsilon = epsilon, 
        regularization=REGULARIZATION)
    train(*args, **kwargs)

if __name__ == "__main__":
    sys.exit(main())
