import argparse
import os, sys
import json
from itertools import izip
from plot_fcns import *
import numpy as np
import pandas as pd
sys.path.insert(0,'/srv/home/nmew/myprojects/clean-neural-nets/')
from common.h5py_loading import load_target_map, align_target_maps, load_target_list
from common.chembl_export_data_loader import DrugMatrixDataLoader
from common.h5py_data_loader import H5pyDataLoader
from lasagne_nn.run_nn import get_predictions_of_knowns, get_network_from_weights

__author__ = "Elena Caceres"
__credits__ = []
__email__ = "ecaceres@keiserlab.org"
"""Generate and store predictions for a neural network experiment."""


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

        
def get_network_script_from_train_path(train_path, network_script_fmter):
    script_name = train_path.split("trained_nets/")[-1].split("/")[0]
    return network_script_fmter.format(script_name)
    
    
class Experiment(dict):
    def __init__(self, name):
        self.name = name
        self.folds = []
        self.converged_epochs = []
        self.trained_paths = []
      
    def __repr__(self):
        return str(vars(self))
    
    def __str__(self):
        return json.dumps(vars(self), indent=2)
        
    def set_converged_epoch(self, epoch, train_path, fold=None):        
        self.folds.append(fold)
        self.converged_epochs.append(epoch)
        self.trained_paths.append(train_path)


def main(hdf5_train, hdf5_ts, target_map_f, outdir, converged_epochs_json, network_script_fmter=None):
    """ Main runs the predictions
    Parameters
    ----------
    hdf5_train : str (*.hdf5)
        training data for the network
    hdf5_ts : str (*.hdf5)
        time split validation for the network
    target_map_f : str (*.pkl)
        target map associating hdf5 matrix position with protein target
    outdir : str
        where to save the data
    converged_epochs_json : str
        json containing list of class Experiment() type objects
    **kwargs
        network_script_fmter : str (default None)
            where the train scripts are stored in format "*/{}.py". If None, one is assigned automatically to nnets base

    """
    print("hdf5 Train: {}".format(hdf5_train))
    print("hdf5 Time Split: {}".format(hdf5_ts))
    print("Target Map File: {}".format(target_map_f))
    print("Outdir: {}".format(outdir))
    print("Converged Epochs Json: {}".format(converged_epochs_json))

    # read converged epochs json file for experiment information.
    with open(converged_epochs_json, "r") as fp:
        expts = json.load(fp)
        
    # this is where we store our 
    if network_script_fmter is None:
        network_script_fmter = "{}/labgits/neural-nets/experiments/{}.py".format(get_env_var("HOME"), "{}")
    # instantiate data loaders for loading and dealing with data
    ts_dl = H5pyDataLoader(hdf5_file=hdf5_ts,
                           target_map_file=target_map_f, 
                           train_percentage=None, 
                           multitask=True)
    ts_dl.load_training_data()
    
    dm_dl = DrugMatrixDataLoader()
    
    train_dl = H5pyDataLoader(hdf5_file=hdf5_train,
                              target_map_file=target_map_f, 
                              train_percentage=None, 
                              multitask=True)

    datasets = ['test', 'train', 'timesplit', 'drugmatrix']

    for e in expts:
        first = True
        print("making predictions for {}".format(e["name"]))
              
        for epoch, path, fold in izip(e["converged_epochs"], e["trained_paths"], e["folds"]):
            print("(fold, epoch, path): ({}, {}, {})".format(fold, epoch, path))
            # epoch network info
            network_script = get_network_script_from_train_path(path, network_script_fmter)
            test_index_file = "{}/test_indices.npy".format(path)
            train_dl.test_indices_file = test_index_file
            weights_f = os.path.join(path, "model_at_epoch_{}.npz".format(epoch))
            network = get_network_from_weights(weights_f, build_nn=network_script)
            # this should update the indices each time.
            train_dl.train_indices, train_dl.test_indices = train_dl.get_train_test_indices()
            train_dl.load_training_data()
            # get data ready for predictions
            # train_known = train_dl.all_act[train_dl.train_indices]
            # test_known = train_dl.all_act[train_dl.test_indices]
            # n_molecules shouldn't change
            # assert(train_known.shape[0] + test_known.shape[0] == train_dl.all_act.shape[0])
            # make predictions
            for ds in datasets:
                preds, knowns = predictions_knowns_from_trained_network_and_data(ds, network, weights_f, train_dl, ts_dl, dm_dl)
                predf = os.path.join(outdir, '{}_{}_{}_regression_preds.npz'.format(e["name"], ds, fold))
                knwnf = os.path.join(outdir, '{}_{}_{}_regression_knowns.npz'.format(e["name"], ds, fold))
                np.savez_compressed(predf, preds)
                np.savez_compressed(knwnf, knowns)
                print("Saved predictions to {}".format(predf))
                print("Saved knowns to {}".format(knwnf))
            # save targets to file
            if first: 
                np.savez('{}/targets/ValTrain_targets.npz'.format(outdir), load_target_list(train_dl.target_map_file))
                # these two should map to the same protein targets
                dm_target_slice, train_target_slice = align_target_maps(dm_dl.target_map, train_dl.target_map)
                np.savez('{}/targets/drugmatrix_targets.npz'.format(outdir), train_target_slice[-1])
                np.savez('{}/targets/timesplit_targets.npz'.format(outdir), load_target_list(ts_dl.target_map_file))
                print("Saved target maps to: {}/targets/*.npz".format(outdir))
                first = False
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(""""Generate predictions for a given nnet.""",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     fromfile_prefix_chars='@')
    parser.add_argument('hdf5_train', type=str,
                        help="""training data for the network (*.hdf5)""")
    parser.add_argument('hdf5_ts', type=str,
                        help="""time split validation for the network (*.hdf5)""")
    parser.add_argument('target_map_f', type=str,
                        help="""target map associating hdf5 matrix position with protein target""")
    parser.add_argument('outdir', type=str,
                    help="""where to save the data""")
    parser.add_argument('converged_epochs_json', type=str,
                help="""json containing list of class Experiment() type objects""")
    parser.add_argument('-n', '--network_script_fmter', type=str, default=None,
                        help="""where the train scripts are stored in format "*/{}.py". If None, one is assigned automatically to nnets base""")
    params = parser.parse_args()
    kwargs = dict(params._get_kwargs())
    hdf5_train = kwargs.pop('hdf5_train')
    hdf5_ts = kwargs.pop('hdf5_ts')
    target_map_f = kwargs.pop('target_map_f')
    outdir = kwargs.pop('outdir')
    converged_epochs_json = kwargs.pop('converged_epochs_json')
    
    print main(hdf5_train, hdf5_ts, target_map_f, outdir, converged_epochs_json, **kwargs)