"""
This script current go-to neural network parameters, we'll update it periodically.
These params produced an r squared of 0.74 on a 20% hold out in 500 epochs when
trained with ChEMBL20 using ecfp4 1024 bit fingerprints.
"""
import os
import sys
import logging
from argparse import ArgumentParser
import lasagne
import lasagne_nn.hdf5_basic_nn as bnn
from lasagne_nn.output_loader import get_weight_files_from_dir
from common.util import ready_dir, config_file_logging, write_args_file
from lasagne_nn.utils import EarlyStopping, StoreNetworkParams, StoreLoss, LogTraining, LogEpoch


BATCH_SIZE = 300  # larger=faster epochs; smaller=better loss/epoch
NUM_EPOCHS = 1000
REGULARIZATION = None
LEARNING_RATE = 0.001
MOMENTUM = 0.5
USE_DROPOUT = True
OUTPUT_ACTIVATION = lasagne.nonlinearities.leaky_rectify
HIDDEN_LAYERS = [
    # (num_units, dropout_p, nonlinearity)
    (512, .10, lasagne.nonlinearities.leaky_rectify),
    (256, .25, lasagne.nonlinearities.leaky_rectify),
    (128, .25, lasagne.nonlinearities.leaky_rectify),
]
NPKI = False
SAVE_EVERY_N_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 550
ON_EPOCH_FINISHED = [StoreLoss(every_n_epochs=100),
                     StoreNetworkParams(every_n_epochs=SAVE_EVERY_N_EPOCHS),
                     LogEpoch(), EarlyStopping(patience=EARLY_STOPPING_PATIENCE)]
ON_TRAINING_FINISHED = [StoreLoss(), StoreNetworkParams(), LogTraining()]
ON_TRAINING_EXCEPTION = [StoreLoss(), StoreNetworkParams()]


def build_nn(input_shape, output_shape, input_var):
    return bnn.build_nn(input_shape, output_shape, input_var,
                        use_dropout=USE_DROPOUT,
                        hidden_layers=HIDDEN_LAYERS,
                        output_activation=OUTPUT_ACTIVATION)


def main():
    parser = ArgumentParser("Train lasagne NN to find targets for given fingerprints", fromfile_prefix_chars='@')
    parser.add_argument('-o', '--output_directory',
                        help="directory where logging, backup and output will get stored")
    parser.add_argument('-d', '--dataset', type=str,
                        help=("file with the dataset (*.hdf5 or *.h5) assumes contains ("
                              "numpy.ndarray) datasets with the names: \n"
                              "\t'activity' : (training_examples,) -- type np.float32\n"
                              "\t'position' : (training_examples,) -- type np.int16\n"
                              "\t'fp_array' : (training_examples, fingerprint_length) -- type "
                              "np.bool"))
    parser.add_argument("--multitask", action="store_true",
                        help="""Whether the input data is multitask or singletask.""")
    parser.add_argument('-e', '--num_epochs', type=int,
                        default=NUM_EPOCHS,
                        nargs='?',
                        help="number of epochs to train")
    parser.add_argument('-i', '--test-index-file', type=str,
                        nargs='?',
                        help='Pickled file of indices to use for testing')
    parser.add_argument('-w', '--weight_file', type=str,
                        default=None,
                        help="npy or pickled weights/network params to initialize network with. "
                             "Start epoch will be based on name of file. eg. model_at_epoch_200.npz")
    parser.add_argument('--log-level', type=str,
                        default='INFO',
                        nargs='?',
                        help='Output log level [default: %(default)s]')
    parser.add_argument("-b", "--batch_size", type=int,
                        default=BATCH_SIZE,
                        nargs='?',
                        help="size of batches to train on [default: %(default)s]")
    parser.add_argument("-p", "--early_stopping_patience", type=int,
                        default=EARLY_STOPPING_PATIENCE,
                        nargs='?',
                        help="number of epochs to consider when deciding to stop training")
    parser.add_argument("--resume", action="store_true",
                        help="""Resume training from existing output directory, chose default test index file,
                        loss files and weight files found in the given output directory.""")
    params, _ = parser.parse_known_args()
    ready_dir(params.output_directory)
    logging.basicConfig(stream=sys.stderr, level=params.log_level)
    config_file_logging(params.output_directory)
    write_args_file(parser, params.output_directory)
    logging.info('Using random seed: {}'.format(bnn.RANDOM_STATE))
    kwargs = dict(training_data=params.dataset,
                  output_dir=params.output_directory,
                  num_epochs=params.num_epochs,
                  multitask=params.multitask,
                  early_stopping_patience=params.early_stopping_patience,
                  build_nn_func=build_nn,
                  learning_rate=LEARNING_RATE,
                  momentum=MOMENTUM,
                  batch_size=params.batch_size,
                  npKi=NPKI,
                  regularization=REGULARIZATION,
                  on_epoch_finished=ON_EPOCH_FINISHED,
                  on_training_finished=ON_TRAINING_FINISHED,
                  on_training_exception=ON_TRAINING_EXCEPTION)

    if params.resume:
        weight_files, epochs = get_weight_files_from_dir(params.output_directory)
        kwargs['weight_file'] = weight_files[-1]
        kwargs['test_errs_file'] = os.path.join(params.output_directory, 'test_loss.csv')
        kwargs['train_errs_file'] = os.path.join(params.output_directory, 'train_loss.csv')
        kwargs['test_indices_filename'] = os.path.join(params.output_directory, 'test_indices.npy')
    if params.weight_file:
        kwargs['weight_file'] = params.weight_file
    if params.test_index_file:
        kwargs['test_indices_filename'] = params.test_index_file

    bnn.train(**kwargs)


if __name__ == '__main__':
    sys.exit(main())
