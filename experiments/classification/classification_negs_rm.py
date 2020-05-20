import sys
import lasagne
import numpy as np
import theano.tensor as T
from common.h5py_data_loader import H5pyDataLoader
from lasagne_nn.hdf5_basic_nn import build_nn as bnn
from lasagne_nn.hdf5_basic_nn import parse_args, train
from lasagne_nn.train import GenericUpdateWithRegularization



LEARNING_RATE = 0.01
MOMENTUM = 0.4
REGULARIZATION = {'lambda': 0.0005, 'penalty': lasagne.regularization.l2}
USE_DROPOUT = True
OUTPUT_ACTIVATION = lasagne.nonlinearities.sigmoid
HIDDEN_LAYERS = [
    # (num_units, dropout_p, nonlinearity, regularization)
    (1024, .10, lasagne.nonlinearities.leaky_rectify),
    (2048, .25, lasagne.nonlinearities.leaky_rectify),
    (3072, .25, lasagne.nonlinearities.leaky_rectify)
]
USE_BATCH_NORM = False
BATCH_NORM_ARGS = None
TRAIN_PERCENTAGE = .8
BINARY_THRESHOLD = 5.0

PERCENT_NEGATIVES_TO_REMOVE = 1.0

class AllNegsSnegsH5pyDataLoader(H5pyDataLoader):
    def __init__(self, negs_to_remove=0., negs_to_remove_negative_thresh=5.0, **kwargs):
        """
        Removes a percentage of known values below a threshold after load_training_data is called.
        :param negs_to_remove: float in range (0., 1.) percent of negatives in dataset to keep (default=1.0)
        :param kwargs: args passed to H5pyDataLoader
        """
        # todo: handle percent of negatives and save indices of removed data
        assert(negs_to_remove == 1.)
        self.negs_to_remove = negs_to_remove
        self.negs_to_remove_negative_thresh = negs_to_remove_negative_thresh
        super(AllNegsSnegsH5pyDataLoader, self).__init__(**kwargs)

    def erase_negatives(self):
        negs_mask = (self.all_act > 0.) & (self.all_act < self.negs_to_remove_negative_thresh)
        self.all_pos[negs_mask] = -1
        self.all_act[negs_mask] = 0.
        try:
            self.all_rel[negs_mask] = ''
        except AttributeError:
            pass

    def load_training_data(self):
        super(AllNegsSnegsH5pyDataLoader, self).load_training_data()
        self.erase_negatives()

class AllNegsSnegsH5pyDataLoader(H5pyDataLoader):
    def __init__(self, negs_to_remove=0., negs_to_remove_negative_thresh=5.0, **kwargs):
        """
        Removes a percentage of known values below a threshold after load_training_data is called.
        :param negs_to_remove: float in range (0., 1.) percent of negatives in dataset to keep (default=1.0)
        :param kwargs: args passed to H5pyDataLoader
        """
        # todo: handle percent of negatives and save indices of removed data
        assert(negs_to_remove == 1.)
        self.negs_to_remove = negs_to_remove
        self.negs_to_remove_negative_thresh = negs_to_remove_negative_thresh
        super(AllNegsSnegsH5pyDataLoader, self).__init__(**kwargs)

    def erase_negatives(self):
        negs_mask = (self.all_act > 0.) & (self.all_act < self.negs_to_remove_negative_thresh)
        self.all_pos[negs_mask] = -1
        self.all_act[negs_mask] = 0.
        try:
            self.all_rel[negs_mask] = ''
        except AttributeError:
            pass

    def load_training_data(self):
        super(AllNegsSnegsH5pyDataLoader, self).load_training_data()
        self.erase_negatives()




def build_data_loader(training_data,
          negs_to_remove=PERCENT_NEGATIVES_TO_REMOVE,
          negs_to_remove_negative_thresh=BINARY_THRESHOLD,
          test_indices_filename=None,
          random_seed=42,
          npKi=False,
          multitask=False,
          stochastic_negatives=False,
          target_map_file=None,
          train_percentage=0.8,
          negative_blacklist_file=None,
          negative_threshold=BINARY_THRESHOLD,
          positive_negative_ratio=None, **kwargs):

    return AllNegsSnegsH5pyDataLoader(
        negs_to_remove=negs_to_remove,
        negs_to_remove_negative_thresh=negs_to_remove_negative_thresh,
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


def build_data_loader(training_data,
          negs_to_remove=PERCENT_NEGATIVES_TO_REMOVE,
          negs_to_remove_negative_thresh=BINARY_THRESHOLD,
          test_indices_filename=None,
          random_seed=42,
          npKi=False,
          multitask=False,
          stochastic_negatives=False,
          target_map_file=None,
          train_percentage=0.8,
          negative_blacklist_file=None,
          negative_threshold=BINARY_THRESHOLD,
          positive_negative_ratio=None, **kwargs):

    return AllNegsSnegsH5pyDataLoader(
        negs_to_remove=negs_to_remove,
        negs_to_remove_negative_thresh=negs_to_remove_negative_thresh,
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


def build_nn(input_shape, output_shape, input_var=None):
    return bnn(input_shape, output_shape,
               input_var=input_var,
               use_dropout=USE_DROPOUT,
               use_batch_norm=USE_BATCH_NORM,
               batch_norm_kwargs=BATCH_NORM_ARGS,
               hidden_layers=HIDDEN_LAYERS,
               output_activation=OUTPUT_ACTIVATION)


class BinaryClassificationTransformer(object):
    def __init__(self, threshold, args_index=1):
        self.threshold = threshold
        self.threshold_t = lasagne.utils.as_theano_expression(lasagne.utils.floatX(self.threshold))
        self.args_index = args_index

    def minibatch_transform(self, *args):
        """Drug Matrix Transformation"""
        return tuple(self.cpu_transform(a) if i == self.args_index else a
                     for i, a in enumerate(args))

    def cpu_transform(self, y):
        return y > self.threshold

    def theano_transform(self, y):
        """theano graph transformation"""
        return T.gt(y, self.threshold_t)

    def objective_transform(self, objective):
        """transform target values to binary before calculating loss"""
        def binary_classification_error(x, y):
            return objective(x, self.theano_transform(y))
        # update name for logging
        binary_classification_error.__name__ = '_'.join(
            [objective.__name__, self.__class__.__name__,
             "threshold_{}".format(self.threshold)])
        return binary_classification_error


def build_train_test(
        learning_rate, momentum, regularization,
        objective=lasagne.objectives.binary_crossentropy,
        binary_threshold=BINARY_THRESHOLD):

    # transform target values to binary before calculating loss
    transformer = BinaryClassificationTransformer(binary_threshold)

    # Masking loss of missing data is handled in GenericUpdateWithRegularization
    return GenericUpdateWithRegularization(
        objective=transformer.objective_transform(objective),
        update=dict(
            function=lasagne.updates.nesterov_momentum,
            kwargs={'learning_rate': learning_rate, 'momentum': momentum}),
        regularization=regularization)


def main():
    args, kwargs = parse_args(learning_rate=LEARNING_RATE, momentum=MOMENTUM,
                              regularization=REGULARIZATION)
    kwargs['build_nn_func'] = build_nn
    regularization = kwargs['build_train_test_func'].regularization
    build_nn_kwargs = kwargs['build_train_test_func'].update['kwargs']
    learning_rate = build_nn_kwargs['learning_rate']
    momentum = build_nn_kwargs['momentum']
    kwargs['build_train_test_func'] = build_train_test(
        learning_rate, momentum, regularization)
    kwargs['data_loader'] = build_data_loader(args[0], **kwargs)
    train(*args, **kwargs)


if __name__ == "__main__":
    sys.exit(main())
