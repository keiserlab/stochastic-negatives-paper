import os
import sys
import time
import shutil
import logging
from itertools import izip
import numpy as np
import h5py
from seacore.run.core import SEACore
from fpcore.fingerprint import fingerprint
from fpcore.fconvert import string2ascii

stream_logging_level = logging.DEBUG
logging.getLogger().setLevel(logging.NOTSET)
sh = logging.StreamHandler(sys.stderr)
sh.setLevel(stream_logging_level)
logging.getLogger().addHandler(sh)

hdf5_file = '/srv/home/nmew/data/hdf5/chembl20/ecfp/with_PCBA/mid_included/time-split/MI_PCBA_subset_chembl20_800maxmw_10moleculesmin_medianval_mid_4096bvECFP_TS2012train.hdf5'
sea_lib = '/srv/home/nmew/myprojects/lab-notebook-mew/notebooks/stochastic_negatives/binding/chembl20_binding_ecfp4_4096.sea'
results_file = 'MI_PCBA_subset_chembl20_800maxmw_10moleculesmin_medianval_mid_4096bvECFP_TS2012train_prediction.csv'
multinstance = True

base_tmp_dir = '/fast/disk1/nmew/blacklist/timesplit'
tmp_dir = os.path.join(base_tmp_dir, 'sea_prediction_{}'.format(time.time()))
tmp_hdf5_file  = os.path.join(tmp_dir, os.path.basename(hdf5_file))
tmp_sea_lib = os.path.join(tmp_dir, os.path.basename(sea_lib))
tmp_results_file = os.path.join(tmp_dir, os.path.basename(results_file))
os.mkdir(tmp_dir)
shutil.copy2(hdf5_file, tmp_hdf5_file)
shutil.copy2(sea_lib, tmp_sea_lib)

print('outputing to {}'.format(tmp_dir))

with h5py.File(tmp_hdf5_file, 'r') as f:
    all_fp = f['fp_array'][()]
    if not multinstance:
        b = np.ascontiguousarray(all_fp).view(np.dtype((np.void, all_fp.dtype.itemsize * all_fp.shape[1])))
        _, idx = np.unique(b, return_index=True)
        unique_fp = all_fp[idx]
    else:
        unique_fp = all_fp
        idx = range(all_fp.shape[0])

unique_fp = unique_fp.astype(np.int8).astype(str)

unique_fp_bit_strings = np.apply_along_axis(lambda x: string2ascii("".join(x)), 
                                            axis=1, 
                                            arr=unique_fp)

fingerprints = [(fp, i) for i, fp in izip(idx, unique_fp_bit_strings)]

core = SEACore()
core.set_outputfile(tmp_results_file)
core.load_reference(tmp_sea_lib)
results = core.run(fingerprints)
print type(results)

core.close()

print('SEA prediction completed and output to {}'.format(tmp_dir))

