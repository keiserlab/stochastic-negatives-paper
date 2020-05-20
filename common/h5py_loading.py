__author__ = 'ecaceres'
import logging
from itertools import cycle
import h5py
import numpy as np
import cPickle as pkl
from common.data_converter import convert_to_pki, convert_scale

TRAIN_PERCENTAGE = 0.8
RANDOM_STATE = 42

def get_train_test_indices(num_training_cases, train_percentage=TRAIN_PERCENTAGE,
                           random_state=RANDOM_STATE):
    """
    Returns train and test indices

    Parameters
    ----------
    num_training_cases : int
        total size of your data to split
    train_percentage : int (default TRAIN_PERCENTAGE)
        percent of training cases that got to train indices
    random_state : int (default RANDOM_STATE)
        random seed
    Returns
    -------
    train_indices : np.ndarray
        indices for the training set
    test_indices : np.ndarray
        indices for the test set
    """
    rng = np.random.RandomState(seed=random_state)
    shuffled_indices = np.arange(num_training_cases)
    rng.shuffle(shuffled_indices)
    train_indices = shuffled_indices[:int(num_training_cases * train_percentage)]
    test_indices = shuffled_indices[int(num_training_cases * train_percentage):]
    return train_indices, test_indices


def load_positive_negative_ratio_map(ratio_map_file):
    """
    Params: File or path to file containing list of targets and their indices (*.pkl)
    Returns: number of targets
    """
    if isinstance(ratio_map_file, str):
        with open(ratio_map_file, 'rb') as rmf:
            ratio_map = pkl.load(rmf)
    else:
        ratio_map = pkl.load(ratio_map_file)
    return ratio_map


def load_target_map(target_map):
    """
    Params: File containing list of targets and their indices (*.pkl)
    Returns: dict with target_ids as keys and indices as values
    """
    if isinstance(target_map, str):
        with open(target_map, 'rb') as target_map_file:
            chembl_map = pkl.load(target_map_file)
    else:
        chembl_map = pkl.load(target_map)
    return chembl_map


def load_target_list(target_map):
    """
    Params: File containing list of targets and their indices (*.pkl)
    Returns: list of target_ids ordered by index
    """
    chembl_map = load_target_map(target_map)
    sorted_chembl_ids = [k for k, v in sorted(chembl_map.items(), key=lambda x: x[1])]
    return sorted_chembl_ids


def load_dataset(hdf5_file, test_indices_file=None, npKi=False, train_percentage=TRAIN_PERCENTAGE,
                           random_state=RANDOM_STATE):
    """
    Load test and train indices of hdf5 file and return
    descriptive data on included datasets

    Parameters
    ----------
    h5data : *.hdf5
        assumes contains (numpy.ndarray) datasets with the names:
            'activity' : (training_examples,) -- type np.float32
            'position' : (training_examples,) -- type np.int16
            'fp_array' : (training_examples, fingerprint_length) -- type np.bool
    test_indices_file : file (OPTIONAL)
        file containing test indices
    npKi (bool):
        whether or not to convert from pKi to a 0-1 scale.
    Returns
    -------
    train_indices : np.ndarray
        indexes referencing the train set
    test_indices : np.ndarray
        indexes referencing the test set
    num_training_cases : int
        number of rows in dataset (number of fp:activity:target groups)
    fp_len : int
        length of fingerprint
    num_targets : int
        number of protein targets in dataset
    """
    with h5py.File(hdf5_file, 'r') as f:
        num_training_cases = f.attrs['training_cases']
        fp_len = f.attrs['fprint_len']
        num_targets = f.attrs['num_targets']
        num_training_cases, fp_len = f['fp_array'].shape
        logging.info("Number of training cases: %d" % num_training_cases)
        logging.info("Fingerprint length: %d" % fp_len)
        logging.info("Number of targets: %d" % num_targets)

        if test_indices_file is not None:
            test_indices = np.load(test_indices_file)
            train_indices = np.delete(np.arange(num_training_cases), test_indices)
        else:
            train_indices, test_indices = get_train_test_indices(num_training_cases,
                                                                 train_percentage=train_percentage,
                                                                 random_state=random_state)

        test_indices = list(test_indices)
        test_indices.sort()
        train_indices = list(train_indices)
        train_indices.sort()

        all_act = f['activity'][()]
        # data should be in nM and then be converted
        all_act = convert_to_pki(all_act)

        if npKi == True:
            # normalize to 0-1 scale if using sigmoid to get an "npKi"
            all_act = convert_scale(all_act, old_min=1.0, old_max=14.0, new_min=0.0, new_max=1.0)

        all_pos = f['position'][()]
        all_fp = f['fp_array'][()]

    return train_indices, test_indices, all_act, all_pos, all_fp, \
            num_training_cases, fp_len, num_targets


def zip_minibatch_indexing(A, B):
    """
    zipped list of A->B when A and B are not of the same length

    Parameters
    ----------
    A : list
        a list of values
    B : list
        a list of values
    Returns
    -------
    zipped list of A->B when A and B may not be of the same length
    """
    return [list(a) for a in (zip(A, cycle(B)) if len(A) > len(B) else zip(cycle(A), B))]


def iterate_minibatches(fp_arr, act_arr, pos_arr, indices, batchsize, num_targets,
                        shuffle=False, multitask=False):
    """
    Gets minibatch of hdf5 formatted data given a set of indices from which to
    randomly sample.

    Parameters
    ----------
    h5data : *.hdf5
        assumes contains (numpy.ndarray) datasets with the names:
            'activity' : (training_examples,) -- type np.float32
            'position' : (training_examples,) -- type np.int16
            'relation' : (training_examples,) -- type S1
            'fp_array' : (training_examples, fingerprint_length) -- type np.bool
    indices : np.ndarray
        indices for your train or test sets from which to sample
    batchsize : int
        size of the minibatch
    num_targets : int
        number of targets to be predicted
    shuffle : bool (default = True)
        shuffle the dataset indices before minibatching?
    multitask : bool (default = False)
        Whether the inputs are multi- or single-task
    Yields
    -------
    x_inputs : np.ndarray (np.float32)
        fingerprint array with minibatch examples
    y_targets : np.ndarray (np.float32)
        target array with -logpKi at one location with minibatch examples
    """
    if shuffle:
        np.random.shuffle(indices)
    if multitask:
        for start_idx in range(0, len(indices), batchsize):
            # indices to grab from
            excerpt = indices[start_idx: start_idx + batchsize]
            # input fingerprint arrays
            x_inputs = fp_arr[excerpt]
            # find indices of known values for batch
            y_row, act_col = np.where(act_arr[excerpt] > 0.)
            act_row = excerpt[y_row]
            # create arrays
            y_targets = np.zeros((len(excerpt), num_targets), dtype=np.float32)
            y_targets[y_row, pos_arr[act_row, act_col]] = act_arr[act_row, act_col]
            assert y_targets.shape[0] == x_inputs.shape[0]
            yield x_inputs.astype(np.float32), y_targets.astype(np.float32), np.asarray(excerpt)
    else:
        for start_idx in range(0, len(indices), batchsize):
            # indices to grab from
            excerpt = list(indices[start_idx: start_idx + batchsize])
            # update array of inputs
            y_targets = np.zeros((len(excerpt), num_targets), dtype=np.float32)
            y_targets[range(len(excerpt)), pos_arr[excerpt]] = act_arr[excerpt]
            # input fingerprint arrays
            x_inputs = fp_arr[excerpt]
            assert y_targets.shape[0] == x_inputs.shape[0]
            yield x_inputs.astype(np.float32), y_targets.astype(np.float32), np.asarray(excerpt)


def align_target_maps(known_target_map, prediction_target_map):
    # reorder & slice prediction to match data-set targets
    sorted_dl_target_id_items = [(tid, i) for tid, i in sorted(known_target_map.items(), key=lambda x: x[1])
                                 if tid in prediction_target_map]
    known_target_slice = (slice(None), [i for tid, i in sorted_dl_target_id_items])
    prediction_target_slice = []
    for tid, ind in sorted_dl_target_id_items:
        prediction_target_slice.append(prediction_target_map[tid])
    prediction_target_slice = (slice(None), prediction_target_slice)

    sorted_known_tids = np.asarray([tid for tid, i in sorted(known_target_map.items(), key=lambda x: x[1])])
    sorted_known_tids = sorted_known_tids[known_target_slice[1]]
    sorted_predi_tids = np.asarray([tid for tid, i in sorted(prediction_target_map.items(), key=lambda x: x[1])])
    sorted_predi_tids = sorted_predi_tids[prediction_target_slice[1]]
    matching_tids = sorted_known_tids == sorted_predi_tids
    if not np.all(matching_tids):
        np.sum(~matching_tids)
        logging.warn("could not align {} of {} targets".format(np.sum(~matching_tids), len(matching_tids)))

    return known_target_slice, prediction_target_slice
