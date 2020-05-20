import sys
import numpy as np
import lasagne
from common.h5py_data_loader import H5pyDataLoader
from lasagne_nn.hdf5_basic_nn import build_nn as bnn
from lasagne_nn.hdf5_basic_nn import parse_args, train


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


LEARNING_RATE = 0.01
MOMENTUM = 0.4
REGULARIZATION = {'lambda': 0.00005, 'penalty': lasagne.regularization.l2}
USE_DROPOUT = True
OUTPUT_ACTIVATION = lasagne.nonlinearities.leaky_rectify
HIDDEN_LAYERS = [
    # (num_units, dropout_p, nonlinearity, regularization)
    (1024, .10, lasagne.nonlinearities.leaky_rectify),
    (2048, .25, lasagne.nonlinearities.leaky_rectify),
    (3072, .25, lasagne.nonlinearities.leaky_rectify)
]
PERCENT_NEGATIVES_TO_REMOVE = 1.0
NEGATIVE_THRESHOLD = 5.0



def build_data_loader(training_data,
          negs_to_remove=PERCENT_NEGATIVES_TO_REMOVE,
          negs_to_remove_negative_thresh=NEGATIVE_THRESHOLD,
          test_indices_filename=None,
          random_seed=42,
          npKi=False,
          multitask=False,
          stochastic_negatives=False,
          target_map_file=None,
          train_percentage=0.8,
          negative_blacklist_file=None,
          negative_threshold=NEGATIVE_THRESHOLD,
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
    return bnn(input_shape, output_shape, input_var=input_var,  use_dropout=USE_DROPOUT,
               hidden_layers=HIDDEN_LAYERS, output_activation=OUTPUT_ACTIVATION)


def main():
    args, kwargs = parse_args(learning_rate=LEARNING_RATE, momentum=MOMENTUM,
                              regularization=REGULARIZATION)
    kwargs['build_nn_func'] = build_nn
    kwargs['data_loader'] = build_data_loader(args[0], **kwargs)
    train(*args, **kwargs)


if __name__ == "__main__":
    sys.exit(main())
