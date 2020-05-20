import os
import sys
import glob
import logging
import argparse
from common.h5py_data_loader import H5pyDataLoader
from common.util import natural_sort_key, config_file_logging, write_args_file, log_repo_and_machine
from common.plots import plot_rsquared
from lasagne_nn.run_nn import get_predictions_of_knowns

__authors__ = ['Nick Mew', 'Elena Caceres']


def main(hdf5_file, weights_filename, build_nn_script, out_dir, test_indices_file=None, npKi=False, multitask=False):
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

    data_loader = H5pyDataLoader(hdf5_file=hdf5_file,
                                 target_map_file=None,
                                 train_percentage=None,
                                 test_indices_file=test_indices_file,
                                 npKi=npKi)

    weight_files = sorted(glob.glob(weights_filename), key=natural_sort_key)
    logging.info("Found {} directories matching {}".format(len(weight_files), weights_filename))

    for weights_file in weight_files:
        model_num = weights_file.split("/")[-1].split(".")[0].split("_")[-1]
        if test_indices_file != None:
            train_title = ("Train set R Squared at Epoch %s" % model_num)
            test_title = ("Test set R Squared %s" % model_num)
            train_save_base = os.path.join(out_dir, "train_set_epoch_{}.png".format(model_num))
            test_save_base = os.path.join(out_dir, "test_set_epoch_{}.png".format(model_num))

            if os.path.exists(train_save_base):
                logging.info("Found existing plot at {}".format(train_save_base))
            else:
                train_set_pred, train_set_known = get_predictions_of_knowns(data_loader, data_loader.train_indices,
                                                                            build_nn_script=build_nn_script)
                plot_rsquared(train_set_pred, train_set_known,
                              title=train_title, img_filename=train_save_base,
                              result_name='train_r2_@{}'.format(model_num))

            if os.path.exists(test_save_base):
                logging.info("Found existing plot at {}".format(test_save_base))
            else:
                test_set_pred, test_set_known = get_predictions_of_knowns(data_loader, data_loader.test_indices,
                                                                          build_nn_script=build_nn_script)
                plot_rsquared(test_set_pred, test_set_known,
                              title=test_title, img_filename=test_save_base,
                              result_name='test_r2_@{}'.format(model_num))
        else:
            my_title = ("All R-squared at Epoch %s" % model_num)
            save_base = os.path.join(out_dir, "rsquared_epoch_{}.png".format(model_num))
            if os.path.exists(save_base):
                logging.info("Found existing plot at {}".format(save_base))
            else:
                set_pred, set_known = get_predictions_of_knowns(data_loader, range(data_loader.num_training_cases),
                                                                build_nn_script=build_nn_script)
                plot_rsquared(set_pred, set_known,
                              title=my_title, img_filename=save_base,
                              result_name='all_r2_@{}'.format(model_num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        """Generate the ChEMBL r-squared plots for test and train sets given a model""")
    parser.add_argument("hdf5_file", type=str,
                        help=""" *.hdf5 file assumes contains numpy.ndarray datasets with:
                            'activity' : (training_examples,) -- type np.float32
                            'position' : (training_examples,) -- type np.int16
                            'fp_array' : (training_examples, fingerprint_length) -- type np.bool""")
    parser.add_argument("out_dir", type=str,
                        help="""Base filename to save data.""")

    parser.add_argument("build_nn_script", type=str,
                        help="""Default script to specify NN params""")

    parser.add_argument("weights_filename", type=str,
                        help="""Model to evaluate data against""")

    parser.add_argument('-i', '--test_indices_file', type=str,
                        default=None,
                        help="""npy or pickled file of indices to use for testing""")

    parser.add_argument("-n", "--npKi", action="store_true",
                        help="""Normalize data assuming npKis were predicted (i.e. sigmoid output
                        layer). Else, will assume pKis were used (ReLU outputs, DEFAULT).""")

    parser.add_argument("--multitask", action="store_true",
                        help="""Whether the input data is multitask or singletask.""")

    params = parser.parse_args()
    kwargs = dict(params._get_kwargs())
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    config_file_logging(params.out_dir)
    log_repo_and_machine()
    write_args_file(parser, params.out_dir)
    hdf5_file = kwargs.pop("hdf5_file")
    out_dir = kwargs.pop("out_dir")
    build_nn_script = kwargs.pop("build_nn_script")
    weights_filename = kwargs.pop("weights_filename")

    main(hdf5_file, weights_filename, build_nn_script, out_dir, **kwargs)
