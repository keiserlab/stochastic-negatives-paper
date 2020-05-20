__author__ = 'Nick'
import logging
import numbers
import numpy as np
import pandas as pd
# min and max vals taken from Chembl20 pKis
MIN_VAL = 1.0
MAX_VAL = 14.0


def convert_scale(old_value, old_min=MIN_VAL, old_max=MAX_VAL, new_max=1, new_min=0):
        """Returns value in new_min to new_max scale
        
        Parameters
        ----------
        old_value : np.float32
            value to convert
        old_min : np.float32 (default SIG_MIN_VAL)
            minimum value in dataset before conversion
        old_max : np.float32 (default SIG_MAX_VAL)
            maximum value in dataset before conversion
        new_max : np.float32 (default 1)
            maximum value after conversion
        new_min : np.float32 (default 0)
            minimum value after conversion
        """
        return ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min


def convert_01_to_pki(compound_target_affinity, min=1, max=14):
    addby = min - 1
    multiplyby = max - addby
    compound_target_affinity *= multiplyby
    if isinstance(compound_target_affinity, pd.DataFrame):
            compound_target_affinity += addby
    elif isinstance(compound_target_affinity, np.ndarray):
        nonzeros = np.nonzero(compound_target_affinity)
        compound_target_affinity[nonzeros] += addby
    else:
        raise TypeError("only numpy ndarray and pandas dataframe types are supported. {} was found"
                        .format(type(compound_target_affinity)))

def convert_to_pki(affinities):
    """Returns value in new_min to new_max scale

    Parameters
    ----------
    affinities : np.float32 or pd.DataFrame
        value to convert (in nanomolar)

    Returns
    ----------
    new_val : np.float32
        new value as pKi
    """

    if isinstance(affinities, pd.DataFrame):
        affinities = -np.log10(affinities)
        affinities += 9
    elif isinstance(affinities, np.ndarray):
        nonzeros = np.nonzero(affinities)
        affinities[nonzeros] = -np.log10(affinities[nonzeros])
        affinities[nonzeros] += 9
    elif isinstance(affinities, numbers.Number):
        affinities = 9 - np.log10(affinities)
    else:
        raise TypeError("only numpy ndarray and pandas dataframe types are supported. {} was found"
                        .format(type(affinities)))
    return affinities


def convert_to_01_distibution(compound_target_affinity):
    if isinstance(compound_target_affinity, pd.DataFrame):
        min = np.min(compound_target_affinity.affinity)
        max = np.max(compound_target_affinity.affinity)
        logging.info('converting to 0 1 dist. MIN={} MAX={}'.format(min, max))
        subtractby = min - 1
        divideby = max - subtractby
        compound_target_affinity.affinity -= subtractby
        compound_target_affinity.affinity /= divideby
    elif isinstance(compound_target_affinity, np.ndarray):
        nonzeros = np.nonzero(compound_target_affinity)
        ctnz = compound_target_affinity[nonzeros]
        min = np.min(ctnz)
        max = np.max(ctnz)
        logging.info('converting to 0 1 dist. MIN={} MAX={}'.format(min, max))
        subtractby = min - 1
        divideby = max - subtractby
        compound_target_affinity[nonzeros] -= subtractby
        compound_target_affinity[nonzeros] /= divideby
        del ctnz
    else:
        raise TypeError("only numpy ndarray and pandas dataframe types are supported. {} was found"
                        .format(type(compound_target_affinity)))
    return compound_target_affinity, min, max
