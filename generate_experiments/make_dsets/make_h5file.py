__author__ = "Elena Caceres"
__email__ = "ecaceres@keiserlab.org"
"""Purpose: generate data for use with training neural networks. DM and PCBA scrubbed"""

import gzip
import argparse
import h5py
import cPickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from collections import defaultdict
from scipy.stats import mode
from common.data_converter import convert_to_pki
np.random.seed(42)

def get_env_var(handle):
    tmp = os.getenv(handle)
    if not tmp:
        raise LookupError("Environment variable: {} not set.".format(handle))
    return tmp.strip("'")


def get_target_mol_pairs(chembl):
    # for ChEMBL
    chembl_df = pd.read_csv(chembl, sep="\t", compression="gzip")
    tmp_df = chembl_df.groupby(["ChEMBL_Target_ID", "ChEMBL_Molecule_ID"]).size().reset_index().rename(columns={0:'count'})
    unique_pairs = {(i.ChEMBL_Target_ID, i.ChEMBL_Molecule_ID) for i in tmp_df.itertuples()}
    return unique_pairs


def run(inchi_map, out_dir, fp_file, chembl_data, output_base_name, cutoff=5.0, fp_len=4096):

    # load inchi:mid map
    print("loading inchi to mid")
    with gzip.open(inchi_map, "rb") as f:
        # store molecules in case we want to go back and look them up.
        ikey_map = pkl.load(f)
    ikey_map = ikey_map.set_index("index").to_dict()[0]

    # load fingerprints:ikey file
    fp = ""
    fp_lookup_dict={ikey_map[i] : np.zeros((fp_len,), dtype=bool) for i in ikey_map.keys()}
    with gzip.open(fp_file, "r") as f:
        for line in f:
            fp, ikey = line.rstrip().split("\t")
            fp_lookup_dict.update({ikey: np.array(map(int, fp), dtype=bool)})
    assert(fp_len == len(np.array(map(int, fp), dtype=bool)))

    print("Fingerprint length: {}".format(fp_len))

    target_mol_pairs = get_target_mol_pairs(chembl_data)

    # unique_pairs = set()
    # for tid, mid in target_mol_pairs:
    #     try:
    #         unique_pairs.update({(tid, ikey_map[mid])})
    #     except KeyError:
    #         print("Skipping MID {} for target {}. Not in INCHI lookup".format(mid, tid))

    print("Length of total tid, mid pairs: {}".format(len(target_mol_pairs)))

    # number of unique molecules/target on on a target basis
    targets_counts = defaultdict(int)
    for pair in target_mol_pairs:
        targets_counts[pair[0]] += 1

    # Get a list of (target, molecule) pairs where the number of unique molecules surpasses 9
    # Re-open our original dataset and get a list of tuples of all (values, relation, doc_id, year) for our data
    all_data = {}
    unique_mid_smiles = {}
    values = []
    df = pd.read_csv(chembl_data, sep="\t", compression="gzip")
    for row in df.iterrows():
        row = row[1]
        doc_id = row["doc_id"]
        year = row["year"]
        tid = row["ChEMBL_Target_ID"]
        mid = row["ChEMBL_Molecule_ID"]
        act = row["Activity (nM)"]
        rel = row["Relation"]
        smi = row["SMILES"]
        if (tid, mid) in target_mol_pairs:
            if rel == '>':
                act = float(act)
                orig_act = act
                # convert to log space
                act = -np.log10(act) + 9
                # add 2-3 logs
                act -= np.random.uniform(2.0, 3.0)
                if act <= 0:
                    # we're in positive space here, people! can't have a negative, that would be madness.
                    if -act > orig_act:
                        act = orig_act
                    else:
                        act = -act
                # and then convert it back to nM space
                act = 10 ** (-(act - 9))
            if (tid, mid) in all_data:
                try:
                    all_data[(tid, mid)].append((int(doc_id), int(year), float(act), rel))
                    values.append(float(act))
                except:
                    all_data[(tid, mid)].append((int(doc_id), None, float(act), rel))
                    values.append(float(act))
            else:
                try:
                    all_data[(tid, mid)] = [(int(doc_id), int(year), float(act), rel)]
                    values.append(float(act))
                except:
                    all_data[(tid, mid)] = [(int(doc_id), None, float(act), rel)]
                    values.append(float(act))
            if mid not in unique_mid_smiles:
                unique_mid_smiles.update({mid: smi})

    # iterate through the values in each pair
    # Used median to reduce the importance of outliers. In the case of an even number, it just takes the mean anyways.
    consensus = {key: None for key in all_data.keys()}

    for key, value in all_data.iteritems():
        # get the median value of binding affinity
        act = np.median(np.asarray([i[2] for i in value]))
        # get the mode of the relation value
        rel = mode(np.asarray([i[3] for i in value]))
        # get the min year
        try:
            year = np.min(np.asarray([i[1] for i in value if i is not None]))
        except:
            year = 0
        consensus[key] = (act, rel[0][0], year)

    # only accept targets with at least 10 positive values
    # indexer
    targets_counts_above_10 = {k: v for (k, v) in targets_counts.items() if v >= 10}
    target_pos_count = {k: 0 for k in targets_counts_above_10}

    # key[0] = target ; key[1] = molecule
    # value [0] = act ; value[1] = rel ; value[2] = year
    for key, value in consensus.items():
        if convert_to_pki(value[0]) > cutoff:
            try: 
                target_pos_count[key[0]] += 1
            except KeyError:
                continue

    final_target_set = {k for k, v in target_pos_count.items() if v >= 10}

    # create lookup table for what positon in the array corresponds to which target
    # ran once, for generation now, use saved target index for consistency
    target_index_outfile = "{}/{}_target_index.pkl".format(out_dir, output_base_name)
    target_index = {}
    count = 0
    
    for target in final_target_set:
        target_index.update({target: count})
        count += 1
        
    with open(target_index_outfile, 'wb') as tout:
        pkl.dump(target_index, tout)

    #discard molecules w no fingerprints
    bad_molecules = {k for k, v in fp_lookup_dict.items() if v.size == 0}
    
    consensus_minus_bad = {k: v for k, v in consensus.items() if k[1] not in bad_molecules and k[0] in target_index}
    print("Bad training cases removed: %d" % (len(consensus) - len(consensus_minus_bad)))
    
    molecule_count={k[1]:0 for k in consensus_minus_bad.keys()}
    for k in consensus_minus_bad.keys(): 
        molecule_count[k[1]] += 1
        
    # get number of instances
    hist_dict = {}
    for v in molecule_count.values():
        if v in hist_dict:
            hist_dict[v] +=1
        else:
            hist_dict[v]=1
            
    fp_len = len(np.array(map(int, fp), dtype=bool))
    num_targets = len(targets_counts_above_10)
    training_cases = len(molecule_count)+3
    max_multi = max(hist_dict.keys())
    
    # make numpy arrays to write
    tmp_fp = np.zeros((training_cases, fp_len), dtype=np.bool)
    tmp_act = np.full((training_cases, max_multi), np.nan, dtype = np.float32)
    tmp_pos = np.full((training_cases, max_multi), np.nan, dtype = np.uint16)
    tmp_rel = np.full((training_cases, max_multi), np.nan, dtype = "S1")
    tmp_year = np.zeros((training_cases, max_multi), dtype = np.uint16)

    # key[0] = target ; key[1] = molecule
    # value [0] = act ; value[1] = rel ; value[2] = year
    keys_sorted_by_mids = sorted(consensus_minus_bad.keys(), key=lambda x: x[1])
    # more accurate: previous key... 
    last_key = keys_sorted_by_mids[0][1]
    row_count = 0
    column_count = 0
    for key in keys_sorted_by_mids:
        value = consensus_minus_bad[key]
        if key[1] == last_key:
            try:
                tmp_fp[column_count] = fp_lookup_dict[ikey_map[key[1]]]
            except KeyError:
                continue
            tmp_act[column_count, row_count] = value[0]
            try:
                tmp_year[column_count, row_count] = value[2]
            except: 
                tmp_year[column_count, row_count] = 0
            tmp_rel[column_count, row_count] = value[1]
            tmp_pos[column_count, row_count] = target_index[key[0]]
            row_count += 1
        else:
            column_count += 1
            row_count = 0
            try:
                tmp_fp[column_count] = fp_lookup_dict[ikey_map[key[1]]]
            except KeyError:
                continue
            tmp_act[column_count, row_count] = value[0]
            try:
                tmp_year[column_count, row_count] = value[2]
            except: 
                tmp_year[column_count, row_count] = 0
            tmp_rel[column_count, row_count] = value[1]
            tmp_pos[column_count, row_count] = target_index[key[0]]

            row_count += 1
            last_key = key[1]
            
    with h5py.File("{}/{}.hdf5".format(out_dir, output_base_name), 'w') as f:
        # pre-allocate arrays for our dataset
        fp_arr = f.create_dataset('fp_array', (training_cases, fp_len), dtype=np.bool, chunks=True, fillvalue=False, compression="lzf")
        act_arr = f.create_dataset('activity', (training_cases, max_multi), dtype=np.float32, chunks=True, fillvalue=0.0, compression="lzf")
        pos_arr = f.create_dataset('position', (training_cases, max_multi), dtype=np.uint16, chunks=True, fillvalue=0, compression="lzf")
        rel_arr = f.create_dataset('rel_array', (training_cases, max_multi), dtype="S1", chunks=True, fillvalue='', compression="lzf")
        year_arr = f.create_dataset('year', (training_cases, max_multi), dtype=np.uint16, chunks=True, fillvalue=0, compression="lzf")

        # set values
        fp_arr[:] = tmp_fp
        act_arr[:] = tmp_act
        pos_arr[:] = tmp_pos
        rel_arr[:] = tmp_rel
        year_arr[:] = tmp_year

        # declare attributes for meta-data
        f.attrs["activity_units"] = "nM, median"
        f.attrs["relationship_type"] = "mode"
        f.attrs["year_type"]="First publication date. If None given, value = 0"
        f.attrs["training_cases"] = training_cases
        f.attrs["num_targets"] = num_targets 
        f.attrs["fprint_len"] = fp_len
        f.attrs["fprint_type"] = "bit/ECFP4"
        f.attrs["desc"] = "Full Training data for ECFP multi-task network with DM scrubbed and no PCBA. 10 positive ligands/target with a cutoff of pac50 of 5.0.  See lookup tables for target indexing"

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(""""make hdf5 files""")
    parser.add_argument("ikey_map", type=str,
                        help="""*.pkl file mapping a molecule id to an inchi key""")
    parser.add_argument("out_dir", type=str,
                        help="""path to save final data""")
    parser.add_argument("fp_file", type=str,
                        help="""fingerprint file containing fingerprints & inchikeys""")
    parser.add_argument("chembl_data", type=str,
                        help="""ChEMBL data to write to file *.csv.gz""")
    parser.add_argument("output_base_name", type=str,
                        help="""base name (without directory or file extension) for output""")
    parser.add_argument('-c', '--cutoff', type=float, default=5.0,
                        help="""pAC50 cutoff for non-binders.""")

    params = parser.parse_args()
    kwargs = dict(params._get_kwargs())
    ikey_map = kwargs.pop("ikey_map")
    out_dir = kwargs.pop("out_dir")
    fp_file = kwargs.pop("fp_file")
    chembl_data = kwargs.pop("chembl_data")
    output_base_name = kwargs.pop("output_base_name")

    run(ikey_map, out_dir, fp_file, chembl_data, output_base_name, **kwargs)
