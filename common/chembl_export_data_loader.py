import os, sys, csv
from itertools import chain
import numpy as np
import pandas as pd
from common.data_loader import DataLoader
from common.data_converter import convert_to_pki
from common.util import get_neural_nets_path as nn_path


class ChemblExportDataLoader(DataLoader):
    def __init__(self, chembl_export_csv, fp_file, fp_col=0, id_col=1, **kwargs):
        super(ChemblExportDataLoader, self).__init__(
            data_file=chembl_export_csv, **kwargs)
        self.chembl_export_csv = chembl_export_csv
        self.fp_file = fp_file
        chembl_export_df = df_from_chembl_export(
            self.chembl_export_csv, fill_unknowns=False).reset_index()
        self.targets = chembl_export_df['target'].unique()
        self.chembl_export_df = pivot_chembl_export_df(chembl_export_df)
        self.fingerprints, self.compounds = load_fingerprint_input(self.fp_file, fp_col=fp_col, id_col=id_col)
        self.compounds = np.asarray(self.compounds)

    @property
    def fp_len(self):
        return self.fingerprints[0].shape[0]

    @property
    def num_training_cases(self):
        return len(self.compounds)

    @property
    def num_targets(self):
        return len(self.targets)

    @property
    def target_map(self):
        return {trgt:i for i, trgt in enumerate(self.targets)}

    def iterate_minibatches(self, batchsize, indices, shuffle=False, include_known_unknowns=False, fill_nan_with=0.00001, **kwargs):
        if indices is None:
            indices = np.arange(self.num_training_cases)
        if shuffle:
            np.random.shuffle(indices)
        if isinstance(batchsize, tuple):
            small, large = batchsize
            if len(indices) < large:
                batchsize = len(indices)
            else:
                batchsize = nearest_divisor_in_range(len(indices), small, large)
            if isinstance(batchsize, tuple):
                batchsize = len(indices) / small
        for start_idx in range(0, len(indices), batchsize):
            excerpt = indices[start_idx: start_idx + batchsize]
            y_targets = self.chembl_export_df.loc[self.compounds[excerpt]].as_matrix(self.targets)
            if include_known_unknowns:
                y_targets[np.isnan(y_targets)] = fill_nan_with
            x_inputs = self.fingerprints[excerpt]
            yield self.apply_transformations(x_inputs.astype(np.float32), y_targets.astype(np.float32), np.asarray(excerpt))

    def get_known_unknowns_mask(self, indices):
        # todo: use chembl's relation column for this instead
        knowns = self.chembl_export_df.loc[self.compounds[indices]].as_matrix(self.targets)
        return np.isnan(knowns)

    def get_known_mask(self, indices):
        knowns = self.chembl_export_df.loc[self.compounds[indices]].as_matrix(self.targets)
        return ~np.isnan(knowns)


class DrugMatrixDataLoader(ChemblExportDataLoader):
    def __init__(self, **kwargs):
        self.DM_PATH = os.path.join(nn_path(), "datasets", "validation_sets", "drug_matrix")
        self.DM_CSV = os.path.join(self.DM_PATH, "drugmatrix_full_chembl20_cutoff800.csv")
        self.DM_FP = os.path.join(self.DM_PATH, "drugmatrix_full_chembl20_cutoff800_4096_fingerprints.fp")
        super(DrugMatrixDataLoader, self).__init__(
            chembl_export_csv=self.DM_CSV,
            fp_file=self.DM_FP, **kwargs)

    def get_known_mask(self, indices):
        return np.ones((len(indices), self.num_targets), dtype=bool)


def pivot_chembl_export_df(chembl_export_df):
    return chembl_export_df.reset_index().pivot(
        index='compound', columns='target', values='affinity')


def unpivot_chembl_export_df(chembl_export_df):
    return pd.DataFrame(
        chembl_export_df.stack(dropna=False),
        columns=['affinity']).swaplevel(0, 1, axis=0).sort_index()


def df_from_chembl_export(export_csv_path, average_duplicates=True, fill_unknowns=True, fill_with=-1., to_pki=True):
    tdf = pd.read_csv(export_csv_path,
                      names=['doc', 'year', 'target', 'compound', 'name', 'affinity', 'relation', 'smiles'],
                      usecols=['target', 'compound', 'affinity'],
                      delimiter='\t',
                      skiprows=[0])
    tdf.set_index(['target', 'compound'], inplace=True)
    tdf.sortlevel(['target', 'compound'], inplace=True)
    # if there are multiple target-compound values, use the mean
    if average_duplicates:
        tdf = pd.DataFrame(tdf.groupby(level=['target', 'compound']).affinity.mean())
    if to_pki:
        tdf.affinity = convert_to_pki(tdf.affinity.values)
    # convert to matrix with nans for missing/unknown values
    tdf = pivot_chembl_export_df(tdf)
    tdf = unpivot_chembl_export_df(tdf)
    if fill_unknowns:
        tdf = tdf.fillna(fill_with)
    return tdf


def load_fingerprint_input(fp_file, fp_col=0, id_col=1, dtype=np.float32):
    """Load fingerprint in csv to numpy array"""
    with open(fp_file, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        fp_id = [(np.asarray(map(dtype, line[fp_col])), line[id_col]) for line in reader]
        fps, ids = zip(*fp_id)
    return np.vstack(fps), ids


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

