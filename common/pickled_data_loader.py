__author__ = 'Nick'
import os
import logging
import numpy as np
import cPickle


def load_data(pickled_fp_file, pickled_cta_file):
    """
    Loads dataset from previously loaded and pickled files.
    :param pickled_fp_file: path to pickled fingerprints file
    :param pickled_cta_file: path to pickled compound target affinities file
    :param logger: logging logger
    """
    logging.info('Loading Data')
    # fingerprints
    fingerprints, compound_ids = load_fingerprints_and_compounds(pickled_fp_file)
    # compound_target_affinities
    compound_target_affinity, target_ids = load_affinities_and_targets(pickled_cta_file)
    return compound_ids, target_ids, fingerprints, compound_target_affinity


def load_custom_pickle(pickled_file):
    with open(pickled_file, 'rb') as f:
        nonzeros = cPickle.load(f)
        nonzero_vals = cPickle.load(f)
        shape = cPickle.load(f)
        ids = cPickle.load(f)
    return nonzeros, nonzero_vals, shape, ids


def load_fingerprints_and_compounds(pickled_fp_file):
    logging.info('Loading fingerprints from: ' + pickled_fp_file)
    fp_nonzeros, fp_nonzero_vals, fp_shape, compound_ids = load_custom_pickle(pickled_fp_file)
    fingerprints = np.zeros(fp_shape, dtype=int)
    fingerprints[fp_nonzeros] = fp_nonzero_vals
    logging.info("fingerprints shape: {}".format(fingerprints.shape))
    return fingerprints.astype(np.float32), compound_ids


def load_affinities_and_targets(pickled_cta_file):
    logging.info('Loading affinities from: ' + pickled_cta_file)
    cta_nonzeros, cta_nonzero_vals, cta_shape, target_ids = load_custom_pickle(pickled_cta_file)
    compound_target_affinity = np.zeros(cta_shape)
    compound_target_affinity[cta_nonzeros] = cta_nonzero_vals
    logging.info('unique targets: {}'.format(len(target_ids)))
    logging.info("associations: {}".format(len(cta_nonzero_vals)))
    return compound_target_affinity.astype(np.float32), target_ids


def load_ids(pickled_data_file):
    with open(pickled_data_file, 'rb') as f:
        _ = cPickle.load(f)  # nonzero indices
        _ = cPickle.load(f)  # nonzero values
        _ = cPickle.load(f)  # shape
        ids = cPickle.load(f)
    del _
    return ids


def get_data_size(pickled_data_file):
    _, _, shape, _ = load_custom_pickle(pickled_data_file)
    del _
    return shape[0]


def load_indices(index_file):
    logging.info('Loading Indices from {}'.format(index_file))
    if not os.path.isfile(index_file):
        logging.error('invalid path to index file')
        raise NotImplementedError()
    else:
        with open(index_file) as f:
            test_indices = cPickle.load(f)
    return test_indices

