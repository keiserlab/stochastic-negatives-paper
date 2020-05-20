import os
import sys
import glob
import logging
import argparse
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
from common.util import (natural_sort_key, config_logging, write_args_file, ready_dir, log_repo_and_machine,
                         get_logfile_and_argfile_from_dir, get_args_from_argfile, get_script_path_from_log)
from common.h5py_loading import load_target_map
from common.h5py_data_loader import H5pyDataLoader
from common.plots import RSquaredPlot, AucPlot, BinaryLabeledMetricsPlot, UnknownsDistributionPlot
from lasagne_nn.run_nn import get_predictions_of_knowns, get_network_from_weights
import numpy as np

__authors__ = ['Nick Mew', 'Elena Caceres']


class PredictAndPlotWhereMissing(object):
    def __init__(self):
        # self.pools = []
        self.pool = ThreadPool(4)

    def close(self):
        self.pool.close()
        self.pool.join()

    def __call__(self, metric_plots, prediction_function):
        """
        Prevent's calling expensive prediction if plots already exist. Gets plot filenames from plot classes and
        only calls prediction function if any of the plots are missing.
        :param metric_plots: list of initialized <common.plots.MetricsPlot>s
        :param prediction_function: function that takes no input and returns (prediction, known)
        """
        missing_plots = [plot for plot in metric_plots if not os.path.exists(plot.get_plot_filename())]
        if missing_plots:
            logging.info("Running network predictions for following plots: \n\t{}".format(
                '\n\t'.join([os.path.basename(plot.get_plot_filename()) for plot in missing_plots])))
            set_prediction, set_known = prediction_function()
            for plot in missing_plots:
                if isinstance(plot, RSquaredPlot):
                    plot.plot(set_prediction, set_known)
                else:
                    self.pool.apply_async(plot.plot, (set_prediction, set_known))


def rsquared_plots_from_trained_network_and_data(hdf5_file, weights_filename, build_nn_script, out_dir,
                                                 set_name=None, test_indices_file=0.5,
                                                 npKi=False, multitask=False,
                                                 network_target_map_file=None,
                                                 dataset_target_map_file=None, pred_thresh=None, regression=True):
    """ Get predictions given a set of indices of interest.
    Args:
        hdf5_file (str):
            *.hdf5 file assumes contains (numpy.ndarray) datasets with the names:
                'activity' : (training_examples,) -- type np.float32
                'position' : (training_examples,) -- type np.int16
                'fp_array' : (training_examples, fingerprint_length) -- type np.bool
        weights_filename (str):
            *.npz file storing the weights at a certain epoch.
            Assumes in the format path/to/dir/model_at_epoch_{}.npz
        build_nn_script (str):
            *.py script used to generate the network in weights_filename
        out_dir (str):
            path/to/dir of where to store figures.

    Kwargs:
        npKi (bool):
            Normalize data assuming npKis were predicted (i.e. sigmoid output layer, True). Else,
            will assume pKis were used (ReLU outputs, False, DEFAULT).

    Output:
        output/dir/test_set_epoch_%s.png (file):
            test set r-squared plot
        output/dir/train_set_epoch_%s.png (file):
            train set r-squared plot
    """
    out_dir = ready_dir(out_dir)
    test_indices_file = None if not test_indices_file else test_indices_file
    data_loader = H5pyDataLoader(hdf5_file=hdf5_file, test_indices_file=test_indices_file,
                                 npKi=npKi, multitask=multitask,
                                 target_map_file=dataset_target_map_file,
                                 train_percentage=None)
    data_loader.load_training_data()

    weight_files = sorted(glob.glob(weights_filename), key=natural_sort_key)
    logging.info("Found {} directories matching {}".format(len(weight_files), weights_filename))
    network = get_network_from_weights(weight_files[0], build_nn=build_nn_script)
    network_target_map = load_target_map(network_target_map_file) if network_target_map_file else None
    predict_and_plot_where_missing = PredictAndPlotWhereMissing()
    for weights_file in weight_files:
        epoch = weights_file.split("/")[-1].split(".")[0].split("_")[-1]
        if test_indices_file:
            # train set
            set_name = "Train"
            train_plots = [RSquaredPlot(set_name, epoch, out_dir)]
            if data_loader.all_rel is not None:
                train_plots.extend([
                    UnknownsDistributionPlot(set_name, epoch, out_dir),
                    BinaryLabeledMetricsPlot(set_name, epoch, out_dir, 5.0, pred_thresh=pred_thresh, regression=regression),
                    BinaryLabeledMetricsPlot(set_name, epoch, out_dir, 6.0, pred_thresh=pred_thresh, regression=regression),
                    AucPlot(set_name, epoch, 5.0, 'tpr-fpr', out_dir),
                    AucPlot(set_name, epoch, 6.0, 'tpr-fpr', out_dir),
                    AucPlot(set_name, epoch, 5.0, 'precision-recall', out_dir),
                    AucPlot(set_name, epoch, 6.0, 'precision-recall', out_dir)])
            train_set_prediction_func = partial(get_predictions_of_knowns,
                                                data_loader=data_loader,
                                                weights_filename=weights_file,
                                                indices=data_loader.train_indices,
                                                network=network,
                                                network_target_map=network_target_map)
            predict_and_plot_where_missing(train_plots, train_set_prediction_func)

            # test set
            set_name = "Test"
            test_plots = [RSquaredPlot(set_name, epoch, out_dir)]
            # if data_loader.all_rel is not None:
            test_plots.extend([
                # UnknownsDistributionPlot(set_name, epoch, out_dir),
                BinaryLabeledMetricsPlot(set_name, epoch, out_dir, 5.0, pred_thresh=pred_thresh, regression=regression),
                BinaryLabeledMetricsPlot(set_name, epoch, out_dir, 6.0, pred_thresh=pred_thresh, regression=regression),
                AucPlot(set_name, epoch, 5.0, 'tpr-fpr', out_dir),
                AucPlot(set_name, epoch, 6.0, 'tpr-fpr', out_dir),
                AucPlot(set_name, epoch, 5.0, 'precision-recall', out_dir),
                AucPlot(set_name, epoch, 6.0, 'precision-recall', out_dir)])
            test_set_prediction_func = partial(get_predictions_of_knowns,
                                               data_loader=data_loader,
                                               weights_filename=weights_file,
                                               indices=data_loader.test_indices,
                                               network=network,
                                               network_target_map=network_target_map)
            predict_and_plot_where_missing(test_plots, test_set_prediction_func)

        else:
            # data-set provided
            set_name = set_name or 'All'
            set_plots = [RSquaredPlot(set_name, epoch, out_dir)]
            if data_loader.all_rel is not None:
                set_plots.extend([
                    # UnknownsDistributionPlot(set_name, epoch, out_dir),
                    BinaryLabeledMetricsPlot(set_name, epoch, out_dir, 5.0, pred_thresh=pred_thresh, regression=regression),
                    BinaryLabeledMetricsPlot(set_name, epoch, out_dir, 6.0, pred_thresh=pred_thresh, regression=regression),
                    AucPlot(set_name, epoch, 5.0, 'tpr-fpr', out_dir),
                    AucPlot(set_name, epoch, 6.0, 'tpr-fpr', out_dir),
                    AucPlot(set_name, epoch, 5.0, 'precision-recall', out_dir),
                    AucPlot(set_name, epoch, 6.0, 'precision-recall', out_dir)])
            set_prediction_func = partial(get_predictions_of_knowns,
                                          data_loader=data_loader,
                                          weights_filename=weights_file,
                                          indices=None,
                                          network=network,
                                          network_target_map=network_target_map)
            predict_and_plot_where_missing(set_plots, set_prediction_func)
    predict_and_plot_where_missing.close()


def add_named_args(parser):
    parser.add_argument('-i', '--test_indices_file', type=str,
                        default=None,
                        help="""npy or pickled file of indices to use for testing""")
    parser.add_argument("-n", "--npKi", action="store_true",
                        help="""Normalize data assuming npKis were predicted (i.e. sigmoid output
                        layer). Else, will assume pKis were used (ReLU outputs, DEFAULT).""")
    parser.add_argument('-t', '--pred_thresh', type=float, default=None,
                        help="Prediction threshold to use. If not set, is set to the true threshold (if regression) or 0.5 (if not regression)")
    parser.add_argument('-r', '--regression', type=bool, default=True,
                        help="Usage: True if regression output, False if classification")
    parser.add_argument("--multitask", action="store_true",
                        help="""Whether the input data is multitask or singletask.""")
    parser.add_argument("--set_name", type=str,
                        help="""Dataset name like 'validation' or 'drugmatrix'. Added to results.csv label,
                        defaults to empty string. Note, should be defined if not passing test_indices_file. """)
    parser.add_argument("--network_target_map_file", type=str, default=None,
                        help="""Target map file for dataset network was trained on""")
    parser.add_argument("--dataset_target_map_file", type=str, default=None,
                        help="""Target map file for given hdf5 file""")
    # Existing Named Args
    # named args with same name as positional args are overwritten by positional args
    # these are here so we can just used named args (like we do when using outputdir
    # followed by named args that override args found in outputdir)
    parser.add_argument("--hdf5_file", type=str,
                        help=""" *.hdf5 file assumes contains numpy.ndarray datasets with:
                                'activity' : (training_examples,) -- type np.float32
                                'position' : (training_examples,) -- type np.int16
                                'fp_array' : (training_examples, fingerprint_length) -- type np.bool""")
    parser.add_argument("--out_dir", type=str,
                        help="""Base directory to save data.""")
    parser.add_argument("--build_nn_script", type=str,
                        help="""Default script to specify NN params""")
    parser.add_argument("--weights_filename", type=str,
                        help="""Model to evaluate data against""")


def args_from_argparser():
    parser = argparse.ArgumentParser(
        """Generate the r-squared plots for test and train sets given a model""")
    parser.add_argument("hdf5_file", type=str,
                        help=""" *.hdf5 file assumes contains numpy.ndarray datasets with:
                                'activity' : (training_examples,) -- type np.float32
                                'position' : (training_examples,) -- type np.int16
                                'fp_array' : (training_examples, fingerprint_length) -- type np.bool""")
    parser.add_argument("out_dir", type=str,
                        help="""Base directory to save data.""")
    parser.add_argument("build_nn_script", type=str,
                        help="""Default script to specify NN params""")

    parser.add_argument("weights_filename", type=str,
                        help="""Model to evaluate data against""")
    add_named_args(parser)
    params = parser.parse_args()
    kwargs = dict(params._get_kwargs())
    config_logging(stream_logging_level=logging.INFO, output_directory=params.out_dir)
    log_repo_and_machine()
    write_args_file(parser, params.out_dir)
    return kwargs


def args_from_train_output_dir():
    output_dir = sys.argv[1]
    config_logging(output_directory=output_dir)
    log_repo_and_machine()
    logging.info('finding arguments in output directory of trained network: {}'.format(output_dir))
    logfile, argfile = get_logfile_and_argfile_from_dir(output_dir)
    kwargs = get_args_from_argfile(argfile, [('-d', '--dataset'), '--npKi', '--multitask'])
    print(kwargs)
    kwargs['test_indices_file'] = os.path.join(output_dir, "test_indices.npy")
    kwargs['hdf5_file'] = kwargs.pop('dataset')
    kwargs['weights_filename'] = os.path.join(output_dir, "model_at_epoch_*.npz")
    kwargs['build_nn_script'] = os.path.join(*get_script_path_from_log(logfile))
    kwargs['out_dir'] = output_dir

    # update with any named args if present
    if len(sys.argv) > 2:
        parser = argparse.ArgumentParser()
        add_named_args(parser)
        defaults = {}
        for action in parser._actions:
            defaults[action.dest] = None    # set all default args in parser to None so we know to ignore them
        parser.set_defaults(**defaults)     # as we don't want defaults to override those found in output dir
        for arg, val in parser.parse_known_args(sys.argv[2:])[0]._get_kwargs():
            if val is not None:
                kwargs[arg] = val

    return kwargs


def main():
    if (len(sys.argv) >= 2 and
            not sys.argv[1].endswith('.hdf5') and
            sys.argv[1] not in ['-h', '--help']):
        kwargs = args_from_train_output_dir()
    else:
        kwargs = args_from_argparser()
    print(kwargs)
    rsquared_plots_from_trained_network_and_data(**kwargs)


if __name__ == "__main__":
    sys.exit(main())

