import os, sys
from glob import glob
import itertools
import json
import re
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0,'/srv/home/nmew/myprojects/clean-neural-nets/')
from common.h5py_loading import load_target_map, load_dataset, align_target_maps
from common.chembl_export_data_loader import DrugMatrixDataLoader
from common.h5py_data_loader import H5pyDataLoader
from common.prediction_analysis import df_from_chembl_export, intersect_truth_prediction
from common.metrics import get_results_from_csv
from lasagne_nn.run_nn import get_predictions_of_knowns, get_network_from_weights
from lasagne_nn.output_loader import df_from_prediction_path
from common.h5py_loading import load_target_list, align_target_maps


def get_env_var(handle):
    ''' Get an environment variable given the handle for the bash variable'''
    tmp = os.getenv(handle)
    if not tmp:
        raise LookupError("Environment variable: {} not set.".format(handle))
    return tmp.strip("'")


def get_result_from_csv_add_fold(result_file):
    fold = 'unknown'
    match = re.search("/fold_[0-9]/", result_file)
    if match:
        fold = match.group(0).strip("/")
    results = get_results_from_csv(result_file)
    for r in results:
        r.update(fold=fold)
    return results


def combine_common_results(results_list):
    res_dict = []
    for fold, rlist in enumerate(results_list):
        for r in rlist:
            r = r.copy()
            res_dict.append(r)
    return pd.DataFrame(res_dict)


def get_best_by_metric(results_df, best_metric, ascending=False, n=1):
    r2_tests_df = results_df[results_df['label'] == best_metric]
    r2_tests_df = r2_tests_df.sort_values(['experiment', 'fold', 'value'], ascending=[True, True, ascending])
    return r2_tests_df.groupby(['experiment', 'fold']).head(n)


def get_converged_epoch(results_df, metrics, top_n=4):
    """Given results dataframe returns dict of dicts with experiment and fold 
    as keys and converged epoch as the value for each fold. 
    top_n : returns the top n results from get_best_by_metric, sorted by ascending"""
    # vote count for converged epoch for each fold in each experiment
    experiment_folds = defaultdict(lambda: defaultdict(Counter))    
    for metric in metrics:
        asc = 'loss' in metric
        for (exp, fold), group in get_best_by_metric(
                results_df, metric, ascending=asc, n=top_n).groupby(['experiment', 'fold']):
            experiment_folds[exp][fold].update(group['epoch'].values.tolist())
    # replace counter with converged epoch
    for experiment_name, fc in experiment_folds.iteritems():
        for fold in fc.iterkeys():
            fc[fold] = fc[fold].most_common(1)[0][0]
    return experiment_folds


def get_convergance_by_vote(results_df, metrics, top_n=4):
    """Returns results dataframe filtered by converged epoch for each experiment and fold."""
    # list of dataframes that only contain converged values
    converged_fold_dfs = []
    converged_epochs = get_converged_epoch(results_df, metrics, top_n=top_n)
    for experiment_name, fold_epochs in converged_epochs.iteritems():
        for fold, converged_epoch in fold_epochs.iteritems():
            converged_fold_dfs.append(
                results_df[(results_df['experiment'] == experiment_name) & 
                           (results_df['fold'] == fold) & 
                           (results_df['epoch'] == converged_epoch)]
            ) 
    return pd.concat(converged_fold_dfs)

def get_converged_epoch_for_trained_network(network_dir, converge_metrics, converge_top_n, return_fold=False):
    rfile = os.path.join(network_dir, "test-train", "results.csv")
    print(rfile)
    results = combine_common_results([get_result_from_csv_add_fold(rfile)])
    results['experiment'] = 'experimentname'
    exp_fold_epoch = get_converged_epoch(results, converge_metrics, top_n=4)
    print(exp_fold_epoch)
    fold, converged_epoch = exp_fold_epoch.popitem()[-1].popitem()
    if return_fold:
        return converged_epoch, fold
    else:
        return converged_epoch
    
# dataset can be 'test', 'train', 'val' or 'drugmatrix'# datase 
def predictions_knowns_from_trained_network_and_data(dataset, network, weights_file, train_dl, ts_dl, dm_dl):
    if dataset == 'test' or dataset == 'train':
        data_loader = train_dl
    if dataset == 'timesplit':
        data_loader = ts_dl
    if dataset == 'drugmatrix':
        data_loader = dm_dl

    network_target_map = load_target_map(train_dl.target_map_file)

    if dataset == 'train':
        km = data_loader.get_known_mask(data_loader.train_indices)
        inds = data_loader.train_indices
    elif dataset == 'test':
        km = data_loader.get_known_mask(data_loader.test_indices)
        inds = data_loader.test_indices
    elif dataset == 'timesplit':
        km = data_loader.get_known_mask(np.arange(len(data_loader.all_pos), dtype=int))
        inds = None
    elif dataset == 'drugmatrix': 
        known_target_slice, _ = align_target_maps(data_loader.target_map, train_dl.target_map)
        km = data_loader.get_known_mask(np.arange(len(data_loader.fingerprints), dtype=int))
        km = km[known_target_slice]
        inds = None
    predictions, knowns = get_predictions_of_knowns(data_loader=data_loader,
                                                    weights_filename=weights_file,
                                                    indices=inds,
                                                    network=network,
                                                    network_target_map=network_target_map)    

    # unravel and save predictions
    pred_matrix = np.zeros(km.shape)
    pred_matrix[:] = np.nan
    pred_matrix[km] = predictions

    # unravel and save knowns
    known_matrix = np.zeros(km.shape)
    known_matrix[:] = np.nan
    known_matrix[km] = knowns

    return pred_matrix, known_matrix
    
class  Experiment(dict):
    def __init__(self, name):
        self.name = name
        self.folds = []
        self.converged_epochs = []
        self.trained_paths = []
      
    def __repr__(self):
        return str(vars(self))
    
    def __str__(self):
        return json.dumps(vars(self), indent=2)
        
    def set_convereged_epoch(self, epoch, train_path, fold=None):        
        self.folds.append(fold)
        self.converged_epochs.append(epoch)
        self.trained_paths.append(train_path)
        


def main():
    myhome = get_env_var("HOME")
    home_save_dir = get_env_var("HOME_SAVE_BASE")
    srv_save_dir = get_env_var("DATA_SAVE_BASE")
    expt_base="CLASSIFIER_SMA_RATIOS_LR03"
    
    sneg_pnrs=[1.0, 1.2222, 0.8182, 1.5, 19.0, 9.0, 4.0, 2.3333, 0.6666, 0.4286, 0.25, 0.1111, 0.0753]
    fold_tmplt = "fold_[0-9]*/pnr_*/"
    
    loss_dir = "{}/output/20180815_Paper_Retrains/trained_nets".format(home_save_dir)
    neg_rm_dir = "{}/lr_nesterov_binary_classifier_1024_2048_3072/".format(loss_dir)

    metrics_for_convergance = ['matthews-corrcoef_binary-5.0_test', 'matthews-corrcoef_binary-6.0_test', 'r2_test']
    experiments = []

    parent_dir = os.path.join(neg_rm_dir, expt_base)
    trained_dirs = glob(os.path.join(parent_dir, fold_tmplt))

    for ratio in sneg_pnrs: 
        print(ratio)
        expt_name = "{}_{}".format(expt_base, ratio)
        print(expt_name)
        exp = Experiment(expt_name)
        trained_dirs_subset = [train_dir for train_dir in trained_dirs if "_{}".format(str(ratio)) in train_dir]
        for train_path in trained_dirs_subset:
            epoch, fold = get_converged_epoch_for_trained_network(
                train_path, metrics_for_convergance, 4, return_fold=True)
            exp.set_convereged_epoch(epoch, train_path, fold)
        experiments.append(exp)
    # network script also consistent with negrm expt
    network_script = '{}/labgits/neural-nets/experiments/classification/lr_nesterov_binary_classifier_1024_2048_3072.py'.format(myhome)

    # datasets
    train_dir = "{}/output/20180815_Paper_Retrains/trained_nets/lr_nesterov_binary_classifier_1024_2048_3072/{}/fold_0/pnr_1.0".format(home_save_dir, expt_base)
    new_timesplit_dir = '{}/20180525_DM_scrubbing/train_data/'.format(srv_save_dir)
    new_timesplit_train = os.path.join(new_timesplit_dir, 'train_ts2012_chembl20_MWmax800_scrubDM_minpos10_cutoff5.hdf5')
    new_timesplit_val = os.path.join(new_timesplit_dir, 'val_ts2012_chembl20_MWmax800_scrubDM_minpos10_cutoff5.hdf5')
    new_timesplit_map = os.path.join(new_timesplit_dir, 'ts2012_chembl20_MWmax800_scrubDM_minpos10_cutoff5_target_index.pkl')

    # trained network
    test_index_file = os.path.join(train_dir, 'test_indices.npy') 
    target_map_file = new_timesplit_map

    # dataset DataLoaders
    ts_dl = H5pyDataLoader(
        hdf5_file=new_timesplit_val,
        target_map_file=new_timesplit_map, 
        train_percentage=None, multitask=True)
    ts_dl.load_training_data()
    dm_dl = DrugMatrixDataLoader()
    train_dl = H5pyDataLoader(
        hdf5_file=new_timesplit_train,
        target_map_file=new_timesplit_map, 
        train_percentage=None, multitask=True)

    train_dl.test_indices_file = os.path.join(train_dir, "test_indices.npy")
    train_dl.train_indices, train_dl.test_indices = train_dl.get_train_test_indices()
    train_dl.load_training_data()

    datasets = ['test', 'train', 'timesplit', 'drugmatrix']
    outdir = "{}/output/20180815_Paper_Retrains/predictions/{}".format(home_save_dir, expt_base)

    print("saving experiments as json to:")
    print(outdir)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    with open(os.path.join(outdir, 'experiments.json'), 'w') as fp:
        json.dump([vars(e) for e in experiments], fp, indent=2)
    first = True
    

    for exp in experiments:
        for i in xrange(len(exp.folds)):
        
            fold = exp.folds[i]
            epoch = exp.converged_epochs[i]
            weights_f = os.path.join(exp.trained_paths[i], 
                                   "model_at_epoch_{}.npz".format(epoch))
            # update test indices for train/test
            testinds_f = os.path.join(exp.trained_paths[i], "test_indices.npy")
            train_dl.test_indices_file = testinds_f
            train_dl.train_indices, train_dl.test_indices = train_dl.get_train_test_indices()
            # get network from weights file
            network = get_network_from_weights(weights_f, build_nn=network_script)
            if not os.path.exists("{}/targets".format(outdir)):
                os.mkdir("{}/targets".format(outdir))
            
            # save targets to file
            if first: 
                np.savez('{}/targets/ValTrain_targets.npz'.format(outdir), load_target_list(train_dl.target_map_file))
                known_target_slice, _ = align_target_maps(dm_dl.target_map, train_dl.target_map)
                np.savez('{}/targets/drugmatrix_targets.npz'.format(outdir), dm_dl.targets[known_target_slice[-1]])
                np.savez('{}/targets/timesplit_targets.npz'.format(outdir), load_target_list(ts_dl.target_map_file))
                first = False
            
            # predict and save
            for ds in datasets:
                preds, knowns = predictions_knowns_from_trained_network_and_data(
                    ds, network, weights_f, train_dl, ts_dl, dm_dl)
                predf = os.path.join(
                    outdir, '{}_{}_{}_regression_preds.npz'.format(exp.name, ds, fold))
                knwnf = os.path.join(
                    outdir, '{}_{}_{}_regression_knowns.npz'.format(exp.name, ds, fold))
                np.savez_compressed(predf, preds)
                np.savez_compressed(knwnf, knowns)
    return

if __name__ == "__main__":
    main()
