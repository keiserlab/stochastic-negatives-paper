from abc import ABCMeta, abstractmethod
import numpy as np
import functools
__author__ = 'Nick'


class DataLoader(object):
    """
    :ivar str file_path: file path to dataset
    :ivar float train_percentage: percent of training cases that go to train indices
    :ivar theano.tensor x_tensor_type:
    :ivar theano.tensor y_tensor_type:
    :ivar numpy.RandomState random_state:
    """
    __metaclass__ = ABCMeta

    def __init__(self, data_file, train_percentage=None,
                 transformations=None,
                 test_indices_file=None,
                 random_seed=None):
        self.data_file = data_file
        self.test_indices_file = test_indices_file
        self.train_percentage = train_percentage or 1.0
        self.random_seed = random_seed
        self.random_state = np.random.RandomState(self.random_seed)
        self.transformations = transformations if transformations is not None else []

    def get_train_test_indices(self):
        """
        Returns train and test indices

        Parameters
        ----------
        num_training_cases : int
            total size of your data to split
        Returns
        -------
        train_indices : np.ndarray
            indices for the training set
        test_indices : np.ndarray
            indices for the test set
        """
        if self.test_indices_file is None:
            train_indices, test_indices = get_train_test_indices(self.num_training_cases,
                                                                 self.train_percentage,
                                                                 self.random_seed)
        else:
            test_indices = np.load(self.test_indices_file)
            train_indices = np.delete(np.arange(self.num_training_cases), test_indices)
        return train_indices, test_indices

    def apply_transformations(self, *output):
        """applies transformations to the iterator's output"""
        return reduce(lambda o, f: f(*o), self.transformations, output)


def get_train_test_indices(num_training_cases, train_percentage, random_state):
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

