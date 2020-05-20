import logging
from collections import namedtuple
from itertools import chain
import numpy as np
import h5py
from common.h5py_loading import load_target_map, iterate_minibatches
from common.data_loader import DataLoader
from common.data_converter import convert_to_pki, convert_scale
__authors__ = 'nmew'

StochasticNegativeConfig = namedtuple('StochasticNegativeConfig',
                                      'negative_blacklist,negative_threshold,positive_negative_ratio,'
                                      'negative_blacklist_file,random_state,distribution_func')


class AbstractDistributionFunc(object):
    def __init__(self, random_state=np.random):
        self.random_state = random_state

    def initialize_random_state(self, random_state):
        self.random_state = random_state
        return self

    def __call__(self, size):
        return np.zeros(size)


class UniformDistributionFunc(AbstractDistributionFunc):
    def __init__(self, random_state=np.random, low=1.5, high=4.5):
        self.random_state = random_state
        self.low = low
        self.high = high
        super(self.__class__, self).__init__(self.random_state)

    def __call__(self, size):
        return self.random_state.uniform(size=size, low=self.low, high=self.high)


class BetaDistributionFunc(AbstractDistributionFunc):
    def __init__(self, random_state=np.random, alpha=3.5, beta=2.0, multiplier=4.5):
        self.random_state = random_state
        self.alpha = alpha
        self.beta = beta
        self.multiplier = multiplier
        super(self.__class__, self).__init__(self.random_state)

    def __call__(self, size):
        return self.random_state.beta(self.alpha, self.beta, size=size) * self.multiplier


class SingleValueDistributionFunc(AbstractDistributionFunc):
    def __init__(self, random_state=None, value=-3.0):
        self.value = value
        super(self.__class__, self).__init__(random_state)

    def __call__(self, size):
        return np.full(size, self.value, np.float32)


class H5pyDataLoader(DataLoader):

    """
    :ivar str file_path: file path to dataset
    :ivar float train_percentage: percent of training cases that go to train indices
    :ivar numpy.RandomState random_state:
    """
    def __init__(self, hdf5_file, target_map_file, train_percentage,
                 test_indices_file=None, random_seed=None, multitask=False, npKi=False,
                 stochastic_negatives=False, negative_blacklist_file=None,
                 negative_threshold=None, positive_negative_ratio=None,
                 stochastic_negative_distribution=UniformDistributionFunc(), **kwargs):
        """
        Parameters
        ----------
        hdf5_file : *.hdf5
            assumes contains (numpy.ndarray) datasets with the names:
                'activity' : (training_examples,) -- type np.float32
                'position' : (training_examples,) -- type np.int16
                'fp_array' : (training_examples, fingerprint_length) -- type np.bool
        test_indices_file : file (OPTIONAL)
            file containing test indices
        npKi (bool):
            whether or not to convert from pKi to a 0-1 scale.
        """
        super(H5pyDataLoader, self).__init__(
            data_file=hdf5_file,
            train_percentage=train_percentage,
            test_indices_file=test_indices_file,
            random_seed=random_seed, **kwargs)
        self.all_pos = None
        self.known_target_indexes = None
        self.single_to_multitask_index_map = None
        self.all_act = None
        self.all_rel = None
        self.all_fp = None
        self.npKi = npKi
        self.multitask = multitask
        self.target_map_file = target_map_file
        self.target_map = load_target_map(self.target_map_file) if self.target_map_file is not None else None
        with h5py.File(self.data_file, 'r') as f:
            self.fp_len = f.attrs['fprint_len']
            self.num_targets = f.attrs['num_targets']
            self.num_training_cases, fp_len = f['fp_array'].shape
            logging.info("Number of training cases: %d" % self.num_training_cases)
            logging.info("Fingerprint length: %d" % self.fp_len)
            logging.info("Number of targets: %d" % self.num_targets)
        self.train_indices, self.test_indices = self.get_train_test_indices()
        self.stochastic_negatives = None
        if stochastic_negatives:
            self.stochastic_negatives = StochasticNegativeConfig(
                negative_blacklist=None,
                negative_blacklist_file=negative_blacklist_file,
                negative_threshold=negative_threshold,
                positive_negative_ratio=positive_negative_ratio,
                distribution_func=stochastic_negative_distribution.initialize_random_state(self.random_state),
                random_state=self.random_state
            )
            self.sample_stochastic_negative_distribution()

    def generate_negative_blacklist(self):
        """Load blacklist (from stochastic_negatives.negative_blacklist_file)
           of predicted positives to prevent setting as stochastic negative"""
        if self.stochastic_negatives and (self.stochastic_negatives.negative_blacklist_file is not None):
            if self.stochastic_negatives.negative_blacklist is None:
                self.stochastic_negatives = self.stochastic_negatives._replace(
                    negative_blacklist=np.zeros((self.num_training_cases, self.num_targets), dtype=bool))
            blacklist_indices = np.genfromtxt(self.stochastic_negatives.negative_blacklist_file,
                                              delimiter=',', dtype=np.int).T
            logging.info("(self.num_training_cases, self.num_targets): ({}, {})".format(self.num_training_cases, self.num_targets))
            logging.info("blacklist_indices.shape: {}".format(blacklist_indices.shape))
            self.stochastic_negatives.negative_blacklist[blacklist_indices[0], blacklist_indices[1]] = True
            logging.info("{} out of {} predictions added to blacklist".format(
                self.stochastic_negatives.negative_blacklist.sum(),
                self.stochastic_negatives.negative_blacklist.shape[0] * self.stochastic_negatives.negative_blacklist.shape[1]))

    def sample_stochastic_negative_distribution(self, sample_size=5000):
        r_dist = self.stochastic_negatives.distribution_func(sample_size)
        bin_count, bin_edges = np.histogram(r_dist)
        sample_data = dict(
            sample_size=sample_size,
            min=np.min(r_dist),
            max=np.max(r_dist),
            avg=np.mean(r_dist),
            histogram_bin_count=bin_count.tolist(),
            histogram_bin_edges=bin_edges.tolist()
        )
        logging.info({'stochastic_negative_distribution_sample': sample_data})
        return sample_data

    def load_relation(self):
        with h5py.File(self.data_file, 'r') as f:
            all_rel = None
            if 'relation' in f:
                all_rel = f['relation'][()]
        return all_rel

    def load_pos(self):
        with h5py.File(self.data_file, 'r') as f:
            all_pos = f['position'][()]
        return all_pos

    def load_activity(self):
        with h5py.File(self.data_file, 'r') as f:
            all_act = f['activity'][()]
            # data should be in nM and then be converted
            all_act = convert_to_pki(all_act)
            if self.npKi:
                # normalize to 0-1 scale if using sigmoid to get an "npKi"
                all_act = convert_scale(all_act, old_min=1.0, old_max=14.0, new_min=0.0, new_max=1.0)

            return all_act

    def load_fingerprints(self):
        with h5py.File(self.data_file, 'r') as f:
            all_fp = f['fp_array'][()]
        return all_fp

    def load_single_to_multi_instance_mapping(self):
        """Load fingerprints, activities into memory and create minibatch generators"""
        logging.debug("loading training data")
        if self.stochastic_negatives is not None and not self.multitask:
            logging.debug("find unique fingerprints...")
            with h5py.File(self.data_file, 'r') as f:
                logging.debug("    load fp")
                all_fp = f['fp_array'][()]
                logging.debug("    reformat fp")
            contiguous_fp = np.ascontiguousarray(all_fp).view(
                np.dtype((np.void, all_fp.dtype.itemsize * all_fp.shape[1])))
            logging.debug("    get uniques")
            # [a,b,c], [3,0,2], [1,1,2,0,2] = np.unique([b,b,c,a,c], return_index=True, return_inverse=True)
            _, unique_inds, single_to_multitask_index_map, counts = np.unique(
                contiguous_fp, return_index=True, return_inverse=True, return_counts=True)
            del all_fp
            del contiguous_fp
            del _
            logging.debug("    argsort")
            index_array = np.argsort(single_to_multitask_index_map)
            logging.debug("    known_target_indexes")
            start = 0

            known_target_indexes = np.empty(len(counts), dtype=object)

            for i, count in enumerate(counts):
                end = start + count
                known_target_indexes[i] = self.all_pos[index_array[start:end]]
                start = end
            logging.debug("    self.known_target_indexes done")
            return known_target_indexes, single_to_multitask_index_map
        else:
            return None, None

    def load_training_data(self):
        self.all_pos = self.load_pos()
        self.known_target_indexes, self.single_to_multitask_index_map = self.load_single_to_multi_instance_mapping()
        self.all_act = self.load_activity()
        self.all_rel = self.load_relation()
        self.all_fp = self.load_fingerprints()
        self.generate_negative_blacklist()

    def get_stochastic_negative_blacklist_mask(self, batch_indices, y_targets):
        """Finds all compound-target knowns given single task indices"""
        logging.debug("get_stochastic_negative_blacklist_mask")
        if self.multitask:
            known_mask = y_targets > 0
            multitask_indices = batch_indices
        else:
            multitask_indices = self.single_to_multitask_index_map[batch_indices]
            known_pos_for_batch_fp = self.known_target_indexes[multitask_indices]
            known_indexes = np.hstack([[idx] * len(v) for idx, v in enumerate(known_pos_for_batch_fp)]).astype(int)
            known_mask = np.zeros(y_targets.shape, dtype=bool)
            known_mask[known_indexes, np.hstack(known_pos_for_batch_fp)] = True
        # add negatives blacklist to known_mask
        if self.stochastic_negatives.negative_blacklist is not None:
            batch_blacklist = self.stochastic_negatives.negative_blacklist[multitask_indices]
            known_mask[batch_blacklist] = True
        return known_mask

    def get_where_known(self, batch_indices):
        # find indices of known values for batch
        y_row, act_col = np.where(self.all_act[batch_indices] > 0.)
        act_row = batch_indices[y_row]
        return y_row, self.all_pos[act_row, act_col]

    def get_known_mask(self, batch_indices):
        known_mask = np.zeros((len(batch_indices), self.num_targets), dtype=bool)
        known_mask[self.get_where_known(batch_indices)] = True
        return known_mask

    def get_known_unknowns_mask(self, batch_indices):
        """Finds all compound-target known unknowns for given indices"""
        logging.debug("get_known_unknowns_mask")
        known_unknowns_mask = np.zeros((len(batch_indices), self.num_targets), dtype=bool)
        if self.all_rel is not None:
            # chembl '>' relations
            batch_gt_rel_indices = np.where(self.all_rel[batch_indices] == '>')
            if self.multitask:
                batch_ku_mol_indices, batch_ku_hdf5_col_indices = batch_gt_rel_indices
                batch_ku_tar_indices = self.all_pos[
                    (batch_indices[batch_ku_mol_indices], batch_ku_hdf5_col_indices)]
            else:
                batch_ku_mol_indices, = batch_gt_rel_indices
                batch_ku_tar_indices = self.all_pos[batch_indices[batch_ku_mol_indices]]

            batch_known_unknown_indices = (batch_ku_mol_indices, batch_ku_tar_indices)
            known_unknowns_mask[batch_known_unknown_indices] = True

        return known_unknowns_mask

    def add_stochastic_negatives(self, x_fingerprints, y_targets, batch_indices):
        """
        Adds stochastic negatives to training data at each minibatch.
        If self.stochastic_negatives is Falsy, returns input.

        :param x_fingerprints: np.array, fingerprint training data
        :param y_targets: np.array, compound target association training data
        :param batch_indices: np.array, indices matching all_fp, all_pos, all_act arrays
        :return: x_fingerprints and y_targets with stochastic negatives added as well as batch_indices
        """
        if self.stochastic_negatives:
            # add stochastic negatives to known unknowns
            known_unknown_mask = self.get_known_unknowns_mask(batch_indices)
            num_known_unknowns = np.sum(known_unknown_mask)
            if num_known_unknowns > 0:
                y_targets[known_unknown_mask] = self.stochastic_negatives.distribution_func(
                    num_known_unknowns)
            # get stochastic negatives for proportion of remaining unknowns
            stochastic_negatives, stochastic_negative_pos = get_stochastic_negatives(
                y_targets,
                self.stochastic_negatives.positive_negative_ratio,
                self.stochastic_negatives.negative_threshold,
                self.stochastic_negatives.distribution_func)
            # add stochastic negatives to unknowns
            known_mask = self.get_stochastic_negative_blacklist_mask(batch_indices, y_targets)
            x_fingerprints, y_targets = stochastically_add_activity_to_unknown_training_data(
                known_mask, x_fingerprints, y_targets,
                activity=stochastic_negatives,
                activity_targets=stochastic_negative_pos,
                random_state=self.stochastic_negatives.random_state,
                is_multitask=self.multitask
            )

        return x_fingerprints, y_targets, batch_indices

    def iterate_minibatches(self, batchsize=(300, 1000), shuffle=False, indices=None,
                            stochastic_negatives=False, include_known_unknowns=True):
        if indices is None:
            indices = np.arange(self.num_training_cases)

        if isinstance(batchsize, tuple):
            small, large = batchsize
            if len(indices) < large:
                batchsize = len(indices)
            else:
                batchsize = nearest_divisor_in_range(len(indices), small, large)
            if isinstance(batchsize, tuple):
                batchsize = len(indices) / small

        """Return the existing h5py minibatch iterator unless snegs is true and self.stochastic_negatives is true."""
        for x, y, indexes in iterate_minibatches(
                multitask=self.multitask,
                fp_arr=self.all_fp,
                act_arr=self.all_act,
                pos_arr=self.all_pos,
                num_targets=self.num_targets,
                indices=indices,
                batchsize=batchsize,
                shuffle=shuffle):
            if stochastic_negatives:
                x, y, indexes = self.add_stochastic_negatives(x, y, indexes)
            if not include_known_unknowns:
                known_unknowns_mask = self.get_known_unknowns_mask(indexes)
                y[known_unknowns_mask] = 0.0
            yield self.apply_transformations(x, y, indexes)

    def iterate_train_minibatches(self, batchsize, shuffle=True, stochastic_negatives=True,
                                  include_known_unknowns=True):
        """iterate over minibatches, yielding minibatch updated with stochastic negatives"""
        for x, y, indexes in self.iterate_minibatches(batchsize=batchsize, shuffle=shuffle, indices=self.train_indices,
                                                      stochastic_negatives=stochastic_negatives,
                                                      include_known_unknowns=include_known_unknowns):
            yield x, y, indexes

    def iterate_test_minibatches(self, batchsize=(7000, 15000), shuffle=False, stochastic_negatives=False,
                                 include_known_unknowns=True):
        """Return the existing h5py minibatch iterator unless snegs is true and self.stochastic_negatives is true."""
        for x, y, indexes in self.iterate_minibatches(batchsize=batchsize, shuffle=shuffle, indices=self.test_indices,
                                                      stochastic_negatives=stochastic_negatives,
                                                      include_known_unknowns=include_known_unknowns):
            yield x, y, indexes


def stochastically_add_activity_to_unknown_training_data(known_mask, x_fingerprints, y_targets,
                                                         activity, activity_targets, random_state, is_multitask):
    """
    Add activity to training inputs.

    :param known_mask: numpy array of size (batch_size, target_count) where known (or presumed) associations are True
    :param x_fingerprints: fingerprint input for training, a numpy array of size (batch_size, fp_size)
    :param y_targets: target association input for training, a numpy array of size (batch_size, target_count)
    :param activity: array containing arrays of values to add to training data
    :param activity_targets: array containing target indexes for array of activities (should be the same length as activity)
    :param random_state: numpy.randam.RandomState
    :param is_multitask: True if multitask
    :return: x_fingerprints, y_targets with activities for random x_fingerprints at given activity_pos added
    """
    # select compounds for stochastic negatives
    if len(activity) == 0:
        # no data to add
        return x_fingerprints, y_targets

    sneg_counts = map(len, activity)
    sneg_size = np.sum(sneg_counts)
    sneg_fp_inds = np.empty(sneg_size, dtype=int)    # row indices for y_targets
    sneg_targ_inds = np.empty(sneg_size, dtype=int)  # col indices for y_targets
    activity = np.hstack(activity)
    index = 0
    for target_index, num_snegs_for_target in zip(activity_targets, sneg_counts):
        if num_snegs_for_target > 0:
            unknowns = ~known_mask[:, target_index]
            num_unknowns = np.sum(unknowns)

            if num_unknowns == 0:
                logging.error("no unknowns found for {} stochastic negatives at target {}".format(
                    num_snegs_for_target, target_index))
            else:
                if num_unknowns < num_snegs_for_target:
                    logging.warn("Only {} unknowns found for {} stochastic negatives at target {}".format(
                        num_unknowns, num_snegs_for_target, target_index))
                sneg_fp_inds[index:index + num_snegs_for_target] = random_state.choice(
                    np.where(~known_mask[:, target_index])[0], num_snegs_for_target)
                sneg_targ_inds[index:index + num_snegs_for_target] = target_index
                index += num_snegs_for_target

    if is_multitask:
        y_targets[sneg_fp_inds, sneg_targ_inds] = activity
        return x_fingerprints, y_targets
    else:
        batch_size = y_targets.shape[0]
        target_size = y_targets.shape[1]
        new_x_targets = np.empty((batch_size + sneg_size, x_fingerprints.shape[1]), dtype=x_fingerprints.dtype)
        new_x_targets[:batch_size] = x_fingerprints
        new_x_targets[batch_size:] = x_fingerprints[sneg_fp_inds]

        # add stochastic negatives to y_targets
        new_y_targets = np.zeros((batch_size + sneg_size, target_size), dtype=y_targets.dtype)
        new_y_targets[:batch_size] = y_targets
        new_y_targets[batch_size:][np.arange(len(sneg_targ_inds)), sneg_targ_inds] = activity
        return new_x_targets, new_y_targets


def get_stochastic_negatives(y_targets, pos_neg_ratio, negative_threshold, distribution_func):
    """
    Calculates how many negatives are needed for each target in y_targets and creates an array of negative values
    for each of targets that needs more. The calculation for how many negatives are needed is based on the difference
    between the actual ratio of positives to negatives for each target in y_targets to the target pos_neg_ratio.

    :param y_targets: matrix with target activity training data where rows are compounds and columns are targets
    :param pos_neg_ratio: ratio of positives to negatives either a np.array with a ratio for each target or a float
    :param negative_threshold: defines what's positive and what's negative
    :param distribution_func: Instance of a <code>DistributionFunc</code>
    :return: tuple (stochastic_negatives, negatives_to_add_pos)
            stochastic_negatives: nd.array where each element is an nd.array of negatives
            negatives_to_add_pos: nd.array containing the target indices for each array of stochastic values in stochastic_negatives
    """
    batch_known_mask = y_targets > 0
    batch_positive_mask = y_targets > negative_threshold
    batch_positive_count = np.sum(batch_positive_mask, axis=0)
    batch_negative_count = np.sum(batch_known_mask & ~batch_positive_mask, axis=0)

    expected_negative_count = np.rint(batch_positive_count / pos_neg_ratio)
    negative_diff = expected_negative_count - batch_negative_count
    negatives_to_add_mask = negative_diff > 0
    negatives_to_add = negative_diff[negatives_to_add_mask]
    negatives_to_add_pos = np.where(negatives_to_add_mask)[0]
    stochastic_negatives = np.array([distribution_func(count) for count in negatives_to_add])
    return stochastic_negatives, negatives_to_add_pos


def nearest_divisor_in_range(num, start, end):
    """Caution: give this a decent sized range for best results"""
    def near_factors(n, range_from, range_to, remainder):
            return sorted(
                set(d for d in
                    chain(*[[i, n//i] for i in range(1, int(min(range_to, n**0.5)) + 1) if n % i == remainder])
                    if range_from <= d <= range_to
                    ))
    divisor = None
    for remain in range(0, 100):
        divisors = near_factors(num, start, end, remain)
        if divisors:
            divisor = divisors[0]
            break
    return divisor
