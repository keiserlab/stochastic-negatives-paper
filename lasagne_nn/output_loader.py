__author__ = 'Nick'
import os
import glob
import logging
import itertools
import numpy as np
import pandas as pd
import theano
from common.util import natural_sort_key
import common.pickled_data_loader as pdl


def get_shared_dataset(data_x, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    return shared_x


def get_weights_from_weightfile(weights_filename):
    """Extract lasagne network weights and parameters from pickled file"""
    with np.load(weights_filename) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    return param_values


def get_weight_files_from_dir(dir, weightfile_prefix='model_at_epoch_', weightfile_suffix='.'):
    """Find all weight/param files in a directory, return list of paths and epochs"""
    pathname = os.path.normpath(dir) + '/' + weightfile_prefix + '*'
    print 'reading from: ' + pathname
    weight_files = sorted(glob.glob(pathname), key=natural_sort_key)
    epochs = [int(s.split(weightfile_prefix)[-1].split(weightfile_suffix)[0]) for s in weight_files]
    return weight_files, epochs


def test_input_from_index_file(test_index_file, fingerprints, compound_ids, affinities):
    test_indices = pdl.load_indices(test_index_file)
    mask = np.zeros(len(compound_ids), dtype=bool)
    mask[test_indices] = True
    fingerprints = get_shared_dataset(fingerprints[mask])
    compound_ids = itertools.compress(compound_ids, mask)
    affinities = affinities[mask]
    return fingerprints, compound_ids, affinities


def load_input(fingerprints_dataset, target_dataset, test_index_file=None, logger=None):
    logging.info('Loading Data')
    compound_ids, target_ids, fingerprints, affinities = pdl.load_data(fingerprints_dataset, target_dataset)
    if test_index_file is not None:
        fingerprints, compound_ids, affinities = test_input_from_index_file(test_index_file,
                                                                            fingerprints,
                                                                            compound_ids,
                                                                            affinities)
    return fingerprints, compound_ids, affinities, target_ids


def _df_from_prediction_file(prediction_file):
    """pandas dataframe from csv containing compound, target pKi predictions"""
    return pd.read_csv(prediction_file,
                       index_col=[0, 1],
                       names=['compound', 'target', 'affinity'],
                       usecols=['compound', 'target', 'affinity'],
                       delimiter=',',
                       skiprows=[0])


def df_from_prediction_path(pathname):
    """given a path or pattern, loads predictions from all matching files into a pandas dataframe"""
    logging.info('loading predictions from:  \n\t{}'.format(pathname))
    prediction_files = sorted(glob.glob(pathname), key=natural_sort_key)
    dfs = [_df_from_prediction_file(prediction_file) for prediction_file in prediction_files]
    df = pd.concat(dfs)
    logging.debug('prediction df size: {}\t\tshape: {}'.format(df.size, df.shape))
    df = df.swaplevel(0, 1, axis=0)
    df.sortlevel(['target', 'compound'], inplace=True)
    return df
