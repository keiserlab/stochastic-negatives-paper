"""
run_nn.py is meant to get a prediction using any neural network defined in lasagne_nn along with lasange neural network
parameters. The neural network script is required to have a build_nn function that returns a lasagne neural network
"""
import os
import sys
import imp
import csv
import glob
import time
import logging
from argparse import ArgumentParser
import numpy as np
import theano
import theano.tensor as T
import lasagne
from common.h5py_loading import load_target_list, align_target_maps
from common.util import config_file_logging, ready_dir, write_args_file
from lasagne_nn.output_loader import get_weights_from_weightfile
import gzip

__authors__ = ['Elena Caceres', 'Nick Mew']

DEFAULT_FP_COL = 0
DEFAULT_ID_COL = 1


class NetworkBuilder(object):
    build_func_dict = {}

    def get_build_func(self, script_filename):
        script_filename = os.path.abspath(script_filename)
        if script_filename not in self.build_func_dict:
            logging.info('loading build_nn function from {}'.format(script_filename))
            subdir = os.path.dirname(script_filename)
            module = '_'.join([os.path.basename(subdir), os.path.splitext(os.path.basename(script_filename))[0]])
            custom_nn = imp.load_source(module, script_filename)
            self.build_func_dict[script_filename] = custom_nn.build_nn
        return self.build_func_dict[script_filename]

nb = NetworkBuilder()


def load_fingerprint_input(fp_file, fp_col=DEFAULT_FP_COL, id_col=DEFAULT_ID_COL, dtype=np.float32):
    """Load fingerprint in csv to numpy array"""
    try: 
        with gzip.open(fp_file, 'rb') as f:
            reader = csv.reader(f, delimiter='\t')
            fp_id = [(np.asarray(map(dtype, line[fp_col])), line[id_col]) for line in reader]
            fps, ids = zip(*fp_id)
    except IOError:
        with open(fp_file, "rb") as f:
            reader = csv.reader(f, delimiter=",")
            fp_id = [(np.asarray(map(dtype, line[fp_col])), line[id_col]) for line in reader]
            fps, ids = zip(*fp_id)
    return np.asarray(fps), ids


def output_prediction(out_file, compound_ids, target_ids, predictions):
    """Write prediction to a file"""
    default_name = 'nn_predicted_targets.csv'
    if os.path.exists(out_file):
        if os.path.isdir(out_file):
            out_file = os.path.normpath(out_file) + "/" + default_name
        else:
            out_file = out_file + '_' + str(time.time())
    logging.info('writing to {}'.format(out_file))
    with open(out_file, 'wb') as of:
        writer = csv.writer(of)
        writer.writerow(('compound', 'target', 'pKi'))
        for cpid, prediction in zip(compound_ids, predictions):
            for tpid, affinity in zip(target_ids, np.nditer(prediction)):
                writer.writerow((cpid, tpid, affinity))
    logging.info('done. wrote prediction to {}'.format(out_file))


def load_stored_weights_to_network(weights_filename, network):
    param_values = get_weights_from_weightfile(weights_filename)
    lasagne.layers.set_all_param_values(network, param_values)


def get_network_from_weights(weights_filename, input_var=None, build_nn=None):
    """ Builds a lasagne network from the script that generated the weights_file, the weights file
    and the example input.

    Args:
        weights_filename (str):
            *.npz file storing the weights at a certain epoch
        input_var (theano.tensor.var.TensorVariable):
            inputs to the network
        build_nn (str):
            either *.py script used to generate the network in weights_filename or the function/callable object

    Returns:
        network (lasagne.layers.dense.DenseLayer):
            the lasagne network to make predictions
    """
    if type(build_nn) is str:
        build_nn = nb.get_build_func(build_nn)
    elif not hasattr(build_nn, '__call__'):
        raise ValueError("build_nn must be of type str or a callable object")

    param_values = get_weights_from_weightfile(weights_filename)
    input_shape = param_values[0].shape[0]
    output_shape = param_values[-1].shape[1] if len(param_values[-1].shape) > 1 else \
        param_values[-1].shape[0]
    input_var = input_var or T.fmatrix("inputs")
    network = build_nn(input_shape, output_shape, input_var)
    lasagne.layers.set_all_param_values(network, param_values)
    return network


def get_prediction_func_from_weights(weights_filename, build_nn=None, network=None):
    """Get a theano function that outputs the prediction of a pretrained network"""
    if build_nn is not None:
        network = get_network_from_weights(weights_filename, build_nn=build_nn)
    elif network is not None:
        lasagne.layers.set_all_param_values(network, get_weights_from_weightfile(weights_filename))
    else:
        raise ValueError("Either build_nn or network must be defined")
    input_var = lasagne.layers.get_all_layers(network)[0].input_var
    prediction = lasagne.layers.get_output(network, deterministic=True)
    prediction_fn = theano.function([input_var], prediction)
    return prediction_fn


def run_nn(inputs, weights_filename=None, build_nn_script=None, lasagne_network=None,
           return_dead_activations=False):
    """Get a prediction from a pretrained network """
    if lasagne_network is None and build_nn_script is None:
        raise ValueError("either lasagne_network or build_nn_script must be defined")
    if lasagne_network is None:
        network = get_network_from_weights(weights_filename, build_nn=build_nn_script)
    else:
        network = lasagne_network

    input_var = lasagne.layers.get_all_layers(network)[0].input_var
    prediction = lasagne.layers.get_output(network, deterministic=True)
    prediction_fn = theano.function([input_var], prediction)
    prediction = prediction_fn(inputs)
    to_return = prediction

    if return_dead_activations:
        from lasagne_nn.utils import get_dead_activations_fn
        dead_activations_fn = get_dead_activations_fn([input_var], network)
        dead_activations = dead_activations_fn(inputs)
        to_return = prediction, dead_activations

    if lasagne_network is None:
        del network  # delete the network if it was created here

    return to_return


def get_predictions_of_knowns(data_loader, indices, weights_filename,
                              network=None, build_nn_script=None,
                              stochastic_negatives=False,
                              include_known_unknowns=True,
                              network_target_map=None):
    """ Get predictions and known values given a set of indices of interest.
    Args:
        data_loader (common.h5py_data_loader.H5pyDataLoader):
            data_loader with data loaded from hdf5 file
        indices (np.array):
            list of indices for which to return predictions
        weights_filename (str):
            *.npz file storing the weights at a certain epoch

    Kwargs:
        network (lasagne.layers.dense.DenseLayer):
            the lasagne network to make predictions (required if build_nn_script not defined)
        build_nn_script (str):
            *.py script used to generate the network in weights_filename (required if network not defined)
    Returns:
        prediction (numpy.ndarray):
            the predicted pKi
        known (numpy.ndarray):
            the known pKi
    """
    first = True
    prediction = np.array([])
    known = np.array([])
    # due to space issues, use iterate minibatches to get predictions
    if build_nn_script is not None:
        prediction_func = get_prediction_func_from_weights(weights_filename, build_nn=build_nn_script)
    elif network is not None:
        prediction_func = get_prediction_func_from_weights(weights_filename, network=network)
    else:
        raise ValueError("either network or build_nn_script must be defined")

    # if network and data-set match, no need to reorder predictions to align with data-set
    known_target_slice = slice(None)
    prediction_target_slice = slice(None)
    if network_target_map and data_loader.target_map:
        # reorder & slice prediction to match data-set targets
        known_target_slice, prediction_target_slice = align_target_maps(data_loader.target_map, network_target_map)

    for inputs, targets, indexes in data_loader.iterate_minibatches(batchsize=(8000, 12000), indices=indices,
                                                                    stochastic_negatives=stochastic_negatives,
                                                                    include_known_unknowns=include_known_unknowns):
        # mask tells us which predictions to assess (1 at a time. Not all v all)
        my_mask = data_loader.get_known_mask(indexes)[known_target_slice]
        if include_known_unknowns:
            targets[data_loader.get_known_unknowns_mask(indexes)] = np.nan
        if first:
            prediction = prediction_func(inputs)[prediction_target_slice][my_mask]
            known = targets[known_target_slice][my_mask]
            first = False
        else:
            prediction = np.hstack((prediction, prediction_func(inputs)[prediction_target_slice][my_mask]))
            known = np.hstack((known, targets[known_target_slice][my_mask]))
    # flatten for use with plotting
    prediction = prediction.ravel()
    known = known.ravel()
    logging.debug(" prediction.shape {}".format(prediction.shape))
    logging.debug(" known.shape {}".format(known.shape))

    return prediction, known


def main():
    parser = ArgumentParser("Run NN to predict targets for given fingerprints", fromfile_prefix_chars='@')
    parser.add_argument('-o', '--output_directory',
                        help="directory where logging and output will get stored")
    parser.add_argument('-f', '--fingerprints_dataset', type=str,
                        help="file with compound ids and fingerprints (*.csv) eg. drugmatrix_chembl20_export.csv")
    parser.add_argument('--fp_col', type=int, nargs='?', default=DEFAULT_FP_COL,
                        help="compound fingerprint column number in fingerprints_dataset")
    parser.add_argument('--id_col', type=int, nargs='?', default=DEFAULT_ID_COL,
                        help="compound id column number in fingerprints_dataset")
    parser.add_argument('-t', '--target_mapping_dataset',
                        help="background target mapping dataset")
    parser.add_argument('-w', '--weights_and_parameters', help="weights/network-parameters file")
    parser.add_argument('-n', '--neural_network_script',
                        help="python script (*.py) that must define a function called build_nn")
    parser.add_argument('--log-level', type=str, default='INFO', nargs='?',
                        help='Output log level [default: %(default)s]')
    params, _ = parser.parse_known_args()
    ready_dir(params.output_directory)
    logging.basicConfig(stream=sys.stderr, level=params.log_level)
    config_file_logging(params.output_directory)
    write_args_file(parser, params.output_directory)
    cp_fps, cp_ids = load_fingerprint_input(params.fingerprints_dataset,
                                            fp_col=params.fp_col,
                                            id_col=params.id_col)
    target_ids = load_target_list(params.target_mapping_dataset)
    thresh = 10000
    weight_files = glob.glob(params.weights_and_parameters)
    for weight_file in weight_files:
        prediction_name = os.path.splitext(os.path.split(weight_file)[1])[0] + '_prediction'
        prediction_func = get_prediction_func_from_weights(weight_file, build_nn=params.neural_network_script)
        if cp_fps.shape[0] > thresh:
            for i in range(0, cp_fps.shape[0], thresh):
                through = min(i + thresh, cp_fps.shape[0] - 1)
                prediction = prediction_func(cp_fps[i:through])
                output_prediction(os.path.join(params.output_directory, prediction_name + 's{}-{}.csv'.format(i, through)),
                                  cp_ids[i:through], target_ids, prediction)
        else:
            prediction_file = os.path.join(params.output_directory, prediction_name + '.csv')
            if not os.path.exists(prediction_file):
                prediction = prediction_func(cp_fps)
                output_prediction(prediction_file, cp_ids, target_ids, prediction)


if __name__ == '__main__':
    sys.exit(main())

