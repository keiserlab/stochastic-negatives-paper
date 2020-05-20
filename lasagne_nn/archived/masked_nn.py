"""
This is an archived version of a neural network with a masked output during training.
See https://github.com/keiserlab/neural-nets/issues/7#issuecomment-149339909
"""
# motivated by http://lasagne.readthedocs.org/en/latest/user/tutorial.html
import os
import sys
import time
import logging
from collections import OrderedDict
from argparse import ArgumentParser

import numpy as np
import theano
import theano.tensor as T
import lasagne

module_path = os.path.realpath(os.path.dirname(__file__))
parent_path = os.path.join(module_path, "..")
sys.path.append(parent_path)
from common.util import config_file_logging, ready_dir, write_args_file
import common.nn_reporter as nnr
import common.pickled_data_loader as pdl
import lasagne_nn.custom_layers as custom_layers


NUM_EPOCHS = 500
TRAIN_PERCENTAGE = .9
BATCH_SIZE = 100  # larger=faster epochs; smaller=better loss/epoch
LEARNING_RATE = 0.1

HIDDEN_LAYERS = [
    # (num_units, dropout_p, nonlinearity)
    (512, .10, lasagne.nonlinearities.rectify),
    (256, .25, lasagne.nonlinearities.rectify),
    (128, .25, lasagne.nonlinearities.rectify),
] # whoakay; rectify >> sigmoid in my testing so far!

USE_DROPOUT = True

# None means it'll seed itself from /de/urandom or equivalent.
# (Note np & lasagne must each be seeded explicitly in code below.)
RANDOM_STATE = 42 #None


def get_train_test_indices(num_compounds):
    train_percentage = TRAIN_PERCENTAGE
    
    rng = np.random.RandomState(seed=RANDOM_STATE)
    shuffled_indices = np.arange(num_compounds)
    rng.shuffle(shuffled_indices)
    train_indices = shuffled_indices[:int(num_compounds * train_percentage)]
    test_indices = shuffled_indices[int(num_compounds * train_percentage):]
    return train_indices, test_indices


def load_dataset(training_data_filenames, test_indices_file=None):
    compound_ids, target_ids, \
    fingerprints, compound_target_affinity = pdl.load_data(*training_data_filenames)
    if test_indices_file is not None:
        test_indices = pdl.load_indices(test_indices_file)
        train_indices = np.delete(np.arange(fingerprints.shape[0]), test_indices)
    else:
        train_indices, test_indices = get_train_test_indices(fingerprints.shape[0])
    X_train = fingerprints[train_indices]
    y_train = compound_target_affinity[train_indices]
    X_val = fingerprints[test_indices]
    Y_val = compound_target_affinity[test_indices]
    return X_train, y_train, X_val, Y_val


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def build_nn(input_shape, output_shape, input_var):
    use_dropout = USE_DROPOUT
    
    # for determinism (https://groups.google.com/forum/#!topic/lasagne-users/85-6gxygtIo)
    lasagne.layers.noise._srng = lasagne.layers.noise.RandomStreams(RANDOM_STATE)
    
    # define input layer
    nnet = lasagne.layers.InputLayer(shape=(None, input_shape),
                                     input_var=input_var)
    # define hidden layer(s)
    for nnodes, dropout, nonlin in HIDDEN_LAYERS:
        nnet = lasagne.layers.DenseLayer(
            nnet, num_units=nnodes,
            nonlinearity=nonlin,
            W=lasagne.init.GlorotUniform())
        if use_dropout:
            nnet = lasagne.layers.DropoutLayer(
                nnet, p=dropout)
    # define output layer
    nnet = custom_layers.MaskedLayer(
        nnet, num_units=output_shape, b=None,
        nonlinearity=lasagne.nonlinearities.rectify)
    # return network
    return nnet


def train(training_data_filenames, output_dir, test_indices_filename=None, learning_rate=LEARNING_RATE,
          num_epochs=NUM_EPOCHS):
    """
    Build and train the neural network
    :param training_data_filenames: tuple with (fingerprint filename, target_association filename)
    :param output_dir: directory where any output files will be written
    :param test_indices_filename: path to file with pickled array of indices to be used for a test holdout set
    :param learning_rate: learning rate (float value)
    :param num_epochs: number of epochs to train on
    """
    # load data
    X_train, y_train, X_val, y_val = load_dataset(training_data_filenames, test_indices_file=test_indices_filename)
    
    # prepare theano variables
    input_var = T.fmatrix('inputs')    # fmatrix b/c it's a batch of float? vectors
    target_var = T.fmatrix('targets')  # fmatrix b/c it's a batch of float vectors
    
    # build nn
    network = build_nn(X_train.shape[1], y_train.shape[1], input_var)

    # get loss and update expressions
    target_indices, = T.sum(target_var, 0).nonzero()
    target_var_data = target_var[:, target_indices]
    loss_mask = T.gt(target_var_data, 0)
    # target_mask ensures we get a prediction for target_var_data (a masked weight is created and stored for later)
    prediction = lasagne.layers.get_output(network, target_mask=target_indices)
    loss = lasagne.objectives.squared_error(prediction, target_var_data)
    loss = lasagne.objectives.aggregate(loss, loss_mask, mode='normalized_sum')

    # using 'trainable' tag, we get the masked weight created during get_output above so this must be called after that
    params = lasagne.layers.get_all_params(network, trainable=True)

    ## note: neterov requires shared variables as params which we don't have, so sgd will have to do for now
    updates = lasagne.updates.sgd(loss, params, learning_rate=learning_rate)

    # update the weights of the output layer with the masked weights we got earlier
    output_weights_update = T.set_subtensor(*updates.popitem(last=True))
    update_items = updates.items()
    update_items.append((network.W, output_weights_update))
    updates = OrderedDict(update_items)

    # compile training function
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # get test/validation
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = lasagne.objectives.aggregate(test_loss, T.gt(target_var, 0), mode='mean')

    ## note: test_acc is ommitted b/c it was with respect to softmax classification

    # compile test/validation function
    val_fn = theano.function([input_var, target_var], test_loss)
    val_errs = []
    train_errs = []

    # loop through epochs
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for inputs, targets in iterate_minibatches(X_train, y_train, BATCH_SIZE, shuffle=True):
            # train one epoch
            train_err += train_fn(inputs, targets)
            train_batches += 1
        train_err = train_err / train_batches
        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        for inputs, targets in iterate_minibatches(X_val, y_val, BATCH_SIZE, shuffle=False):
            err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1
        val_err = val_err / val_batches
        # Then we print the results for this epoch:
        logging.info("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        logging.info("  training loss:\t\t{:.9f}".format(train_err))
        logging.info("  validation loss:\t\t{:.9f}".format(val_err))
        train_errs.append(train_err)
        val_errs.append(val_err)

        # Optionally, you could now dump the network weights to a file like this:
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            np.savez(output_dir + '/model_at_epoch_{}.npz'.format(epoch),
                     *lasagne.layers.get_all_param_values(network))

    np.savetxt(os.path.join(output_dir, 'train_errors.csv'), np.asarray(train_errs), delimiter=',')
    np.savetxt(os.path.join(output_dir, 'test_errors.csv'), np.asarray(val_errs), delimiter=',')
    nnr.plot_error(train_errs, title="Train Errors", filename=os.path.join(output_dir, 'train_errors.png'))
    nnr.plot_error(val_errs, title="Test Errors", filename=os.path.join(output_dir, 'test_errors.png'))


def main():
    parser = ArgumentParser("Train lasagne NN to find targets for given fingerprints", fromfile_prefix_chars='@')
    parser.add_argument('-o', '--output_directory', required=True,
                        help="directory where logging, backup and output will get stored")
    parser.add_argument('-f', '--fingerprints_dataset', type=str,
                        help="file with compound ids and fingerprints (*.csv)")
    parser.add_argument('-t', '--target_dataset', type=str,
                        help="file with target id, target chembl id, affinity and list of binding compounds (*.csv)")
    parser.add_argument('-e', '--num_epochs', type=int,
                        default=NUM_EPOCHS,
                        nargs='?',
                        help="number of epochs to train")
    parser.add_argument('--test-index-file', type=str,
                        nargs='?',
                        help='Pickled file of indices to use for testing')
    parser.add_argument('--log-level', type=str,
                        default='INFO',
                        nargs='?',
                        help='Output log level [default: %(default)s]')
    params, _ = parser.parse_known_args()
    print params.output_directory
    ready_dir(params.output_directory)
    logging.basicConfig(stream=sys.stderr, level=params.log_level)
    config_file_logging(params.output_directory)
    write_args_file(parser, params.output_directory)
    logging.info('Using random seed: {}'.format(RANDOM_STATE))
    train((params.fingerprints_dataset, params.target_dataset), params.output_directory, num_epochs=params.num_epochs)


if __name__ == '__main__':
    sys.exit(main())
