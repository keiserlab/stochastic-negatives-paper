"""
functions used in both training and running a network or any that could be used in multiple training functions
"""
__author__ = 'Nick'
import os
import logging
import warnings
from string import capwords
from datetime import timedelta
import numpy as np
import theano
import theano.tensor as T
import lasagne
from common.plots import plot_error
from lasagne_nn.output_loader import get_weights_from_weightfile

WEIGHT_FILE_PATTERN = "model_at_epoch_{}.npz"


def get_epoch_from_weight_file(filename, weight_file_pattern=WEIGHT_FILE_PATTERN):
    prefix, suffix = weight_file_pattern.split('{}')
    return int(filename.split(prefix)[-1].split(suffix)[0])


def store_network_params(output_dir, network, epoch, weight_file_pattern=WEIGHT_FILE_PATTERN):
    weight_file_path = os.path.join(output_dir, weight_file_pattern.format(epoch))
    logging.info("saving network weights to {}".format(weight_file_path))
    np.savez(weight_file_path, *lasagne.layers.get_all_param_values(network))


def load_network_params(file_path, network):
    param_values = get_weights_from_weightfile(file_path)
    lasagne.layers.set_all_param_values(network, param_values)


def store_errors(output_dir, errors_array, file_name, plot_title):
    np.savetxt(os.path.join(output_dir, file_name + '.csv'), np.asarray(errors_array, np.float), delimiter=',')
    plot_error(errors_array, title=plot_title, filename=os.path.join(output_dir, file_name + '.png'))


def load_errors(file_path):
    return list(np.loadtxt(file_path, delimiter=','))


def replace_handler(on_event_functions, handler_class, **kwargs):
    """
    Replace handler of a given type in an existing list of handler instances
    with an new handler of same type with different initializations. For example,
    replace the default EarlyStopping function with one that has more patience.
    on_epoch_end = [LogEpoch(), EarlyStopping(patience=200)]
    replace_handler(on_epoch_end, EarlyStopping, patience=500)
    :param on_event_functions: list of handler instances
    :param handler_class: handler class like EarlyStopping
    :param kwargs: arguments to pass to handler initialization
    """
    for index, func in enumerate(on_event_functions):
        if isinstance(func, handler_class):
            on_event_functions[index] = handler_class(**kwargs)


class LossHistory(list):
    # noinspection PyMissingConstructor
    def __init__(self, name, loss=None):
        self.name = name
        self.display_name = capwords(name.replace('_', ' '))
        if loss is not None:
            self.extend(loss)


class TrainHistory(object):
    def __init__(self, output_dir=None, epoch=None, start_time=None, end_time=None, losses=None):
        self.output_dir = output_dir or './'
        self.losses = losses or []
        # self.train_loss = train_loss or []
        # self.test_loss = test_loss or []
        # self.val_loss = val_loss or []
        self.epoch = epoch or []
        self.start_time = start_time or []
        self.end_time = end_time or []
        self.latest_weights = None

    def get_loss_by_name(self, name):
        for loss in self.losses:
            if loss.name == name:
                return loss
        return None

    def record_epoch(self, epoch=None, start_time=None, end_time=None, losses=None):
        # self.latest_weights = network.get_all_params_values()
        if losses is None:
            losses = []
        for loss_history in self.losses:
            if loss_history.name in losses:
                loss_history.append(losses[loss_history.name])
            else:
                loss_history.append(None)
        # self.train_loss.append(train_loss)
        # self.test_loss.append(test_loss)
        # self.val_loss.append(val_loss)
        self.epoch.append(epoch)
        self.start_time.append(start_time)
        self.end_time.append(end_time)


class LogTraining(object):
    def __call__(self, nn, train_history):
        try:
            logging.info("Total training time took {} (h:m:s)".format(
                timedelta(seconds=train_history.end_time[-1] - train_history.start_time[0])))
        except IndexError:
            logging.warn("Can't log total training time, insufficient records")


class LogEpoch(object):
    def __init__(self, every_n_epochs=1):
        self.every_n_epochs = every_n_epochs

    def __call__(self, nn, train_history):
        if train_history.epoch[-1] % self.every_n_epochs == 0:
            logging.info("Epoch {} took {:.3f}s".format(
                train_history.epoch[-1], train_history.end_time[-1] - train_history.start_time[-1]))
            for loss in train_history.losses:
                logging.info("  {}: {:.9f}".format(loss.display_name, loss[-1]))
            # logging.info("  test loss:     {:.9f}".format(train_history.test_loss[-1]))


class StoreNetworkParams(object):
    def __init__(self, every_n_epochs=1):
        self.every_n_epochs = every_n_epochs

    def __call__(self, nn, train_history):
        if train_history.epoch[-1] % self.every_n_epochs == 0:
            store_network_params(train_history.output_dir, nn, train_history.epoch[-1])


class StoreLoss(object):
    def __init__(self, every_n_epochs=1):
        self.every_n_epochs = every_n_epochs

    def __call__(self, nn, train_history):
        if train_history.epoch[-1] % self.every_n_epochs == 0:
            for loss in train_history.losses:
                store_errors(train_history.output_dir, loss, loss.name, loss.display_name)
            # store_errors(train_history.output_dir, train_history.test_loss, 'test_loss', 'Test Loss')


class EarlyStopping(object):
    def __init__(self, patience=5, min_epoch=25, stop_iteration=False, loss_name='test_loss'):
        self.patience = patience
        self.min_epoch = min_epoch
        self.stop_iteration = stop_iteration
        self.loss_name = loss_name
        self.best_loss = np.inf
        self.best_loss_epoch = 0
        self.best_loss_weights = None
        self.stored_epochs = []
        self.no_loss_warned = False

    def __call__(self, nn, train_history):
        if train_history.get_loss_by_name(self.loss_name) is None:
            # if loss by that name doesn't exist, warn once
            if not self.no_loss_warned:
                warnings.warn("EarlyStopping: No loss '{}' found in TrainHistory. ".format(self.loss_name) +
                              "There will be no early stopping using this loss.")
                self.no_loss_warned = True
            return
        current_loss = train_history.get_loss_by_name(self.loss_name)[-1]
        current_epoch = train_history.epoch[-1]
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_loss_epoch = current_epoch
            self.best_loss_weights = lasagne.layers.get_all_param_values(nn)
        elif current_epoch >= self.min_epoch and self.best_loss_epoch + self.patience < current_epoch and self.best_loss_epoch not in self.stored_epochs:
            logging.info("Best {} in last {} epochs was {:.6f} at epoch {}.".format(
                self.loss_name, self.patience, self.best_loss, self.best_loss_epoch))
            tmp_weights = lasagne.layers.get_all_param_values(nn)
            lasagne.layers.set_all_param_values(nn, self.best_loss_weights)
            store_network_params(train_history.output_dir, nn, self.best_loss_epoch)
            lasagne.layers.set_all_param_values(nn, tmp_weights)
            store_network_params(train_history.output_dir, nn, train_history.epoch[-1])
            self.stored_epochs.append(self.best_loss_epoch)
            if self.stop_iteration:
                logging.info("Early stopping.")
                raise StopIteration()


def get_dead_activations(network):
    """
    Returns a list. For each layer (except the input layer) returns a 1d tensor of type int8 and
    size number of nodes in layer where values are 1 if the node is dead and 0 otherwise.
    :param network: lasagne network that's been built with an input var
    :return: list of arrays, one for each dense layer with 1 signifying a dead activation, 0 otherwise
    """
    all_layers = lasagne.layers.get_all_layers(network)
    layers = [layer for layer in all_layers if isinstance(layer, lasagne.layers.DenseLayer)]
    predictions = lasagne.layers.get_output(layers, deterministic=True)
    dead_activations = [T.eq(T.sum(prediction, 0), 0.) for prediction in predictions]
    return dead_activations


def get_dead_activations_fn(input_vars, network):
    """
    dead_activations_fn = get_dead_activations_fn([input_var], network)
    :param input_vars: python-list of input vars
    :param network:
    :return:
    """
    return theano.function(input_vars, get_dead_activations(network))


def count_dead_activations(all_dead_activations, num_batches):
    for activations in all_dead_activations:
        max_dead = np.max(activations)
        if max_dead > num_batches:
            warnings.warn("more dead activations found than number of batches. Found {}, num_batches is {}".format(
                max_dead, num_batches), RuntimeWarning)
    all_dead_activations = [layer_da >= num_batches for layer_da in all_dead_activations]
    layer_percent_dead = [np.mean(layer_da) for layer_da in all_dead_activations]
    percent_dead = 0
    total_activs = 0
    for activations in all_dead_activations:
        percent_dead += np.sum(activations)
        total_activs += activations.shape[0]
    percent_dead /= total_activs
    return all_dead_activations, layer_percent_dead, percent_dead
