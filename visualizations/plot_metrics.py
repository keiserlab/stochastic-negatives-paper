import os
import abc
import six
from common.util import (natural_sort_key, config_logging, write_args_file, ready_dir, log_repo_and_machine,
                         get_logfile_and_argfile_from_dir, get_args_from_argfile, get_script_path_from_log)
from common.h5py_loading import load_dataset
from common.h5py_data_loader import H5pyDataLoader
from common.metrics import label_zones, label_binary, label_bins
from common.plots import plot_rsquared, plot_labeled_metrics, plot_binary_auc
from lasagne_nn.run_nn import get_predictions_of_knowns, get_network_from_weights

author = 'Nick Mew'

def predict_and_plot_where_missing(plot_classes, prediction_function):
    missing_plots = [not os.path.exists(plot.get_plot_filename()) for plot in plot_classes]
    if any(missing_plots):
        set_prediction, set_known = prediction_function()
        for plot in plot_classes:
            if os.path.exists(plot.get_plot_filename()):
                plot.plot(set_prediction, set_known)
