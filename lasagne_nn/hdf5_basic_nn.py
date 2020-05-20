# motivated by http://lasagne.readthedocs.org/en/latest/user/tutorial.html
import os
import sys
import time
import glob
import pprint
import shutil
import logging
import numpy as np
import theano.tensor as T
import lasagne
from argparse import ArgumentParser
from common.h5py_data_loader import H5pyDataLoader
from common.util import config_file_logging, ready_dir, write_args_file, log_repo_and_machine
from lasagne_nn.train import GenericUpdateWithRegularization
from lasagne_nn.output_loader import get_weight_files_from_dir
from lasagne_nn.utils import load_errors, load_network_params, get_epoch_from_weight_file, replace_handler,\
    LossHistory, TrainHistory, EarlyStopping, StoreNetworkParams, StoreLoss, LogTraining, LogEpoch
__authors__ = ['ecaceres', 'nmew']

TRAIN_PERCENTAGE = .8
BATCH_SIZE = 300  # larger=faster epochs; smaller=better loss/epoch
NUM_EPOCHS = 300
LEARNING_RATE = 0.1
MOMENTUM = 0.5
REGULARIZATION = {'lambda': 0.000001, 'penalty': lasagne.regularization.l2}
UPDATE_FUNCTION = lasagne.updates.nesterov_momentum
HIDDEN_LAYERS = [
    # (num_units, dropout_p, nonlinearity, regularization)
    (512, .10, lasagne.nonlinearities.leaky_rectify),
    (256, .25, lasagne.nonlinearities.leaky_rectify),
    (128, .25, lasagne.nonlinearities.leaky_rectify),
]

OUTPUT_ACTIVATION = lasagne.nonlinearities.leaky_rectify
USE_DROPOUT = True

NEGATIVE_THRESHOLD = 5.0

# None means it'll seed itself from /de/urandom or equivalent.
# (Note np & lasagne must each be seeded explicitly in code below.)
RANDOM_STATE = 42  # None

STORE_LOSS_EVERY_N_EPOCHS = 10
SAVE_EVERY_N_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 50
ON_TRAINING_FINISHED = [StoreLoss(), StoreNetworkParams(), LogTraining()]
ON_TRAINING_EXCEPTION = [StoreLoss(), StoreNetworkParams()]


def build_on_epoch_finished_handlers(store_loss_every_n_epochs=STORE_LOSS_EVERY_N_EPOCHS,
                                     store_weights_every_n_epochs=SAVE_EVERY_N_EPOCHS,
                                     early_stopping_patience=EARLY_STOPPING_PATIENCE,
                                     early_stopping_loss_names=('test_loss', 'test_sneg_loss', 'test_no_gt_loss')):
    ofhs = [StoreLoss(every_n_epochs=store_loss_every_n_epochs),
            StoreNetworkParams(every_n_epochs=store_weights_every_n_epochs),
            LogEpoch()]
    for loss_name in early_stopping_loss_names:
        ofhs.append(EarlyStopping(patience=early_stopping_patience, loss_name=loss_name))
    return ofhs


def build_train_test(learning_rate=LEARNING_RATE, momentum=MOMENTUM, regularization=REGULARIZATION):
    return GenericUpdateWithRegularization(
        objective=lasagne.objectives.squared_error,
        update=dict(
            function=lasagne.updates.nesterov_momentum,
            kwargs={'learning_rate': learning_rate, 'momentum': momentum}),
        regularization=regularization)


def build_nn(input_shape, output_shape, input_var=None, use_dropout=USE_DROPOUT,
             hidden_layers=HIDDEN_LAYERS, output_activation=OUTPUT_ACTIVATION,
             use_batch_norm=False, batch_norm_kwargs=None, random_state=RANDOM_STATE):
    # log params
    logging.info(pprint.pformat(locals()))

    if input_var is None:
        input_var = T.fmatrix('inputs')

    # for determinism (https://groups.google.com/forum/#!topic/lasagne-users/85-6gxygtIo)
    lasagne.layers.noise._srng = lasagne.layers.noise.RandomStreams(random_state)

    # define input layer
    nnet = lasagne.layers.InputLayer(shape=(None, input_shape),
                                     input_var=input_var)
    # define hidden layer(s)
    for nnodes, dropout, nonlin in hidden_layers:
        nnet = lasagne.layers.DenseLayer(
            nnet, num_units=nnodes,
            nonlinearity=nonlin,
            W=lasagne.init.GlorotUniform())
        if use_batch_norm:
            batch_norm_kwargs = batch_norm_kwargs or {}
            nnet = lasagne.layers.batch_norm(nnet, **batch_norm_kwargs)
        if use_dropout:
            nnet = lasagne.layers.DropoutLayer(nnet, p=dropout)
    # define output layer
    nnet = lasagne.layers.DenseLayer(
        nnet, num_units=output_shape,
        nonlinearity=output_activation)
    # return network
    return nnet


def train(training_data, output_dir,
          num_epochs=NUM_EPOCHS,
          test_indices_filename=None,
          batch_size=BATCH_SIZE,
          build_nn_func=build_nn,
          build_train_test_func=build_train_test(),
          data_loader=None,
          random_seed=RANDOM_STATE,
          npKi=False,
          multitask=False,
          weight_file=None,
          loss_files=None,
          stochastic_negatives=False,
          target_map_file=None,
          train_percentage=TRAIN_PERCENTAGE,
          negative_blacklist_file=None,
          negative_threshold=None,
          positive_negative_ratio=None,
          on_epoch_finished=build_on_epoch_finished_handlers(),
          on_training_finished=ON_TRAINING_FINISHED,
          on_training_exception=ON_TRAINING_EXCEPTION):
    """
    Build and train the neural network
    :param build_train_test_func:
    :param training_data: *.hdf5 file
    :param output_dir: directory where any output files will be written
    :param num_epochs: number of epochs to train on
    :param test_indices_filename: file with test indices
    :param batch_size: size of mini-batches to train
    :param build_nn_func: function that builds build_nn_func
    :param npKi: true to train with pKi converted to 0-1 space (False by default)
    :param multitask: whether the input is multitask or not
    :param weight_file: path to existing weight_file
    :param loss_files: list of paths to existing train and test loss csv files
    :param on_epoch_finished: list of functions/callables that are called after an epoch has completed
    :param on_training_finished: list of functions/callables that are called after training has completed
    :param on_training_exception: list of functions/callables that are called if an exception occurs during training
    """
    # log params
    logging.info(pprint.pformat(locals()))

    # load data
    if not data_loader:
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
            positive_negative_ratio=positive_negative_ratio
        )

    data_loader.load_training_data()

    # save target map/index file
    if target_map_file is not None:
        shutil.copy(target_map_file, output_dir)
    # save test indices
    new_test_indices_filename = os.path.abspath(os.path.join(output_dir, "test_indices.npy"))
    if test_indices_filename is None or os.path.abspath(test_indices_filename) != new_test_indices_filename:
        np.save(new_test_indices_filename, data_loader.test_indices)

    # build nn
    network = build_nn_func(data_loader.fp_len, data_loader.num_targets)
    if weight_file is not None:
        load_network_params(weight_file, network)
        start_epoch = get_epoch_from_weight_file(weight_file) + 1
    else:
        start_epoch = 0

    # build train and test functions
    train_fn, test_fn = build_train_test_func(network)

    # loop through epochs
    losses = dict(
        train_loss=LossHistory('train_loss', [None] * start_epoch),
        test_loss=LossHistory('test_loss', [None] * start_epoch)
    )
    if stochastic_negatives:
        losses['test_sneg_loss'] = LossHistory('test_sneg_loss', [None] * start_epoch)
    if data_loader.all_rel is not None:
        losses['test_no_gt_loss'] = LossHistory('test_no_gt_loss', [None] * start_epoch)

    loss_files = [] if loss_files is None else loss_files
    for loss_file in loss_files:
        loss_name = os.path.splitext(os.path.split(loss_file)[-1])[0]
        losses[loss_name] = LossHistory(loss_name, load_errors(loss_file)[:start_epoch])

    train_hist = TrainHistory(
        output_dir=output_dir,
        losses=losses.values(),
        epoch=range(start_epoch)
    )
    begin_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        train_loss = 0
        train_batches = 0
        test_loss = 0
        test_batches = 0
        test_sneg_loss = 0
        test_sneg_batches = 0
        test_no_gt_loss = 0
        test_no_gt_batches = 0
        start_time = time.time()
        losses = {}
        try:
            # In each epoch, we do a full pass over the training data:
            for inputs, targets, _ in data_loader.iterate_train_minibatches(batch_size):
                train_loss += train_fn(inputs, targets)  # train one batch
                train_batches += 1
            train_loss = train_loss / train_batches
            losses['train_loss'] = train_loss

            # And a full pass over the test data:
            for inputs, targets, _ in data_loader.iterate_test_minibatches(stochastic_negatives=False):
                err = test_fn(inputs, targets)
                test_loss += err
                test_batches += 1
            test_loss = test_loss / test_batches
            losses['test_loss'] = test_loss

            # And a full pass over the test data with stochastic negatives:
            if stochastic_negatives:
                for inputs, targets, _ in data_loader.iterate_test_minibatches(stochastic_negatives=True):
                    err = test_fn(inputs, targets)
                    test_sneg_loss += err
                    test_sneg_batches += 1
                test_sneg_loss = test_sneg_loss / test_sneg_batches
                losses['test_sneg_loss'] = test_sneg_loss

            # And a full pass over the test data with no gt relations:
            if data_loader.all_rel is not None:
                for inputs, targets, _ in data_loader.iterate_test_minibatches(stochastic_negatives=False,
                                                                               include_known_unknowns=False):
                    err = test_fn(inputs, targets)
                    test_no_gt_loss += err
                    test_no_gt_batches += 1
                test_no_gt_loss = test_no_gt_loss / test_no_gt_batches
                losses['test_no_gt_loss'] = test_no_gt_loss

            # record epoch and call on epoch finished functions
            train_hist.record_epoch(
                epoch=epoch,
                losses=losses,
                start_time=start_time,
                end_time=time.time()
            )
            for func in on_epoch_finished:
                func(network, train_hist)

        except StopIteration:
            break
        except KeyboardInterrupt:
            logging.warn('KeyboardInterrupt at epoch {}'.format(epoch))
            if train_hist.epoch[-1] != epoch:
                losses['train_loss'] = np.nan if train_batches < (
                    (data_loader.train_indices.shape[0] / batch_size) - 2) else train_loss / train_batches
                losses['test_loss'] = np.nan if test_batches < (
                    (data_loader.test_indices.shape[0] / batch_size) - 2) else test_loss / test_batches
                if stochastic_negatives:
                    losses['test_sneg_loss'] = np.nan if test_sneg_batches < (
                        (data_loader.test_indices.shape[0] / batch_size) - 2) else test_sneg_loss / test_sneg_batches
                for loss_name, loss_value in losses.items():
                    if loss_value == np.nan:
                        logging.warn('{} not calculated for epoch {}'.format(loss_name, epoch))
                train_hist.record_epoch(
                    epoch=epoch,
                    losses=losses,
                    start_time=start_time,
                    end_time=time.time()
                )

            stop = raw_input("Are you sure you want to stop training? "
                             "\n\tYes, store the current state and stop training this network. [Y]"
                             "\n\tNo, it was an accident and I'm sorry (this epoch will be skipped): [hit enter] ")
            if stop.strip() == 'Y':
                break
            else:
                logging.info("Continuing training.")
        except:
            if epoch > 0:
                logging.error('encountered error at epoch {}. Storing network params and errors'.format(epoch))
                for func in on_training_exception:
                    func(network, train_hist)
            raise

    # Training complete, call on training finished functions
    for func in on_training_finished:
        func(network, train_hist)


def parse_args(argv=None, random_state=RANDOM_STATE, early_stopping_patience=EARLY_STOPPING_PATIENCE,
               num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, momentum=MOMENTUM,
               regularization=REGULARIZATION, negative_threshold=NEGATIVE_THRESHOLD, train_percentage=TRAIN_PERCENTAGE,
               store_loss_every_n_epochs=STORE_LOSS_EVERY_N_EPOCHS, store_weights_every_n_epochs=SAVE_EVERY_N_EPOCHS):
    parser = ArgumentParser("Train lasagne NN to find targets for given fingerprints",
                            fromfile_prefix_chars='@')

    parser.add_argument("-o", "--output_directory", required=True,
                        help="directory where logging, backup and output will get stored")

    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help=("file with the dataset (*.hdf5 or *.h5) assumes contains ("
                              "numpy.ndarray) datasets with the names: \n"
                              "\t'activity' : (training_examples,) -- type np.float32\n"
                              "\t'position' : (training_examples,) -- type np.int16\n"
                              "\t'fp_array' : (training_examples, fingerprint_length) -- type "
                              "np.bool"))

    parser.add_argument('-t', '--target_map_file', type=str,
                        default=None,
                        help='pickled file of target ids and their indexes to use for predictions')

    parser.add_argument("-e", "--num_epochs", type=int,
                        default=num_epochs,
                        nargs='?',
                        help="number of epochs to train [default: %(default)s]")

    parser.add_argument("-b", "--batch_size", type=int,
                        default=batch_size,
                        nargs='?',
                        help="size of batches to train on [default: %(default)s]")

    parser.add_argument("-l", "--learning_rate", type=float,
                        default=learning_rate,
                        nargs='?',
                        help="learning rate [default: %(default)s]")

    parser.add_argument("-m", "--momentum", type=float,
                        default=momentum,
                        nargs='?',
                        help="momentum [default: %(default)s]")
    if regularization:
        parser.add_argument("-r", "--regularization_lambda", type=float,
                            default=regularization['lambda'],
                            nargs='?',
                            help="lambda or regularization coefficient [default: %(default)s]")

    parser.add_argument("-p", "--early_stopping_patience", type=int,
                        default=early_stopping_patience,
                        nargs='?',
                        help="number of epochs to consider when deciding to stop training")

    parser.add_argument("-n", "--npKi", action="store_true",
                        help="""Normalize data assuming npKis were predicted (i.e. sigmoid output
                        layer). Else, will assume pKis were used (ReLU outputs, DEFAULT).""")

    parser.add_argument("-w", "--weight_file", type=str,
                        default=None,
                        help="npy or pickled weights/network params to initialize network with. "
                             "Start epoch will be based on name of file. eg. model_at_epoch_200.npz")

    parser.add_argument("-i", "--test_index_file", type=str,
                        default=None,
                        help="npy or pickled file of indices to use for testing")

    parser.add_argument("--train_percentage", type=float,
                        default=train_percentage,
                        help="if test_index_file is not provided, "
                             "defines how much of the dataset is used for training vs testing")

    parser.add_argument("--stochastic_negatives", action="store_true",
                        help="""Resume training from existing output directory, chose default test index file,
                        loss files and weight files found in the given output directory.""")

    parser.add_argument('--negative_blacklist_file', type=str,
                        default=None,
                        nargs='?',
                        help="list of predicted positives to prevent setting as stochastic negative")

    parser.add_argument('--positive_negative_ratio', type=float,
                        default=None,
                        nargs='?',
                        help="npy or pickled weights/network params to initialize network with. "
                             "Start epoch will be based on name of file. eg. model_at_epoch_200.npz")

    parser.add_argument('--negative_threshold', type=float,
                        default=negative_threshold,
                        nargs='?',
                        help="Float value to use as positive/negative threshold")

    parser.add_argument("--store_weights_every_n_epochs", type=int,
                        default=store_weights_every_n_epochs,
                        nargs='?',
                        help="Store weights to file every n number of epochs.")

    parser.add_argument("--store_loss_every_n_epochs", type=int,
                        default=store_loss_every_n_epochs,
                        nargs='?',
                        help="Store loss to output file every n number of epochs.")

    parser.add_argument("--random_state", type=int,
                        default=random_state,
                        nargs='?',
                        help="Number to use as the random seed.")

    parser.add_argument("--log-level", type=str,
                        default="INFO",
                        nargs='?',
                        help="Output log level [default: %(default)s]")

    parser.add_argument("--multitask", action="store_true",
                        help="""Whether the input data is multitask or singletask.""")

    parser.add_argument("--resume", type=str, nargs="?", dest="resume", const="",
                        help="""Resume training from existing output directory, chose default test index file,
                        loss files and weight files automatically. If no value passed, resume from output directory.
                        Otherwise, value is assumed to be a directory and will resume from that directory.""")

    params, _ = parser.parse_known_args(argv)
    ready_dir(params.output_directory)
    logging.basicConfig(stream=sys.stderr, level=params.log_level)
    config_file_logging(params.output_directory)
    log_repo_and_machine()
    write_args_file(parser, params.output_directory)
    logging.info('Using random seed: {}'.format(random_state))
    if regularization:
        regularization['lambda'] = params.regularization_lambda
    kwargs = dict(
        random_seed=params.random_state,
        target_map_file=params.target_map_file,
        num_epochs=params.num_epochs,
        batch_size=params.batch_size,
        train_percentage=params.train_percentage,
        npKi=params.npKi,
        multitask=params.multitask,
        build_train_test_func=build_train_test(
            learning_rate=params.learning_rate,
            momentum=params.momentum,
            regularization=regularization),
        on_epoch_finished=build_on_epoch_finished_handlers(
            store_loss_every_n_epochs=params.store_loss_every_n_epochs,
            store_weights_every_n_epochs=params.store_weights_every_n_epochs,
            early_stopping_patience=params.early_stopping_patience
        ))
    if params.resume is not None:
        if params.resume == "":
            resume_dir = params.output_directory
        else:
            resume_dir = params.resume
        weight_files, epochs = get_weight_files_from_dir(resume_dir)
        kwargs['weight_file'] = weight_files[-1]
        kwargs['loss_files'] = glob.glob(os.path.join(resume_dir, '*_loss.csv'))
        kwargs['test_indices_filename'] = os.path.join(resume_dir, 'test_indices.npy')
    if params.weight_file:
        kwargs['weight_file'] = params.weight_file
    if params.test_index_file:
        kwargs['test_indices_filename'] = params.test_index_file
    if params.stochastic_negatives:
        kwargs['stochastic_negatives'] = True
        kwargs['positive_negative_ratio'] = params.positive_negative_ratio
        kwargs['negative_threshold'] = params.negative_threshold
        kwargs['negative_blacklist_file'] = params.negative_blacklist_file

    return [params.dataset, params.output_directory], kwargs


def main(argv=sys.argv[1:]):
    args, kwargs = parse_args(argv)
    train(*args, **kwargs)


if __name__ == "__main__":
    sys.exit(main())
