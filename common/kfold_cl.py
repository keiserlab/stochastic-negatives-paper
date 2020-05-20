from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import subprocess
import sys
import logging
import cPickle
logger = logging.getLogger(__name__)
from sklearn import cross_validation
from common.pickled_data_loader import get_data_size as get_pickled_data_size
from common.h5py_data_loader import H5pyDataLoader
from common.util import write_args_file, ready_dir, config_file_logging

CREATE_SCREEN = False
RUN_SYNCHRONIZED = True
DIR_PATTERN = "{fold_dir}"
INDEX_PATTERN = "{index_file}"


def get_data_size(dataset_file):
    if dataset_file.endswith('hdf5'):
        dl = H5pyDataLoader(hdf5_file=dataset_file, target_map_file=None, train_percentage=None)
        return dl.num_training_cases
    else:
        return get_pickled_data_size(dataset_file)
    

def create_shell_script(command, path):
    logger.info("creating shell script with your args " + path)
    with open(path, 'w') as shf:
        shf.write(command)
    os.chmod(path, 0755)


def run_shell_script(path, create_screen, screen_name):
    call_args = ["screen", "-d", "-m", "-S", screen_name, "./" + path, '&']
    if create_screen:
        logger.info("creating screen '" + screen_name + "' with script: " + path)
        subprocess.call(call_args)
    else:
        logger.info("run shell script in a screen like so: \n\t" + " ".join(call_args) + " \n")


def k_fold_caller(output_folder, dataset, n_folds, script_args=(), data_size=None,
                  create_screen=CREATE_SCREEN,
                  run_synchronized=RUN_SYNCHRONIZED,
                  dir_pattern=DIR_PATTERN,
                  index_pattern=INDEX_PATTERN):
    logger.info('Loading Data')
    if data_size is None or data_size < 1:
        data_size = get_data_size(dataset)
    kf = cross_validation.KFold(data_size, n_folds=n_folds, shuffle=True)

    output_dir = ready_dir(output_folder)
    pickled_filenames = []
    for k, (train, test) in enumerate(kf):
        pickled_filenames.append(output_dir + '/pickleKF_' + str(k) + '_indices')
        kf_file = open(pickled_filenames[-1], 'wb')
        cPickle.dump(test, kf_file, -1)
        kf_file.close()

    # call via os
    logger.info(
        "Call the script with the following {} files that contain pickled training indices:\n{}".
        format(n_folds, '\n'.join(pickled_filenames)))

    if script_args:
        screen_name_prefix = "_kfold_screen_"
        commands = []
        shell_scripts = []
        for k, index_file in enumerate(pickled_filenames):
            fold_dir = os.path.join(output_dir, "fold_" + str(k))
            commands.append(" ".join(script_args).replace(dir_pattern, fold_dir).replace(index_pattern, index_file))

        for k, command in enumerate(commands):
            screen_name = screen_name_prefix + str(k)
            script_path = os.path.join(output_dir, screen_name + ".sh")
            create_shell_script(command, script_path)
            if not run_synchronized:
                run_shell_script(script_path, create_screen, screen_name)
            else:
                shell_scripts.append(script_path)

        if run_synchronized:
            command = './' + '; ./'.join(shell_scripts)
            screen_name = screen_name_prefix + "0-" + str(len(pickled_filenames))
            script_filename = os.path.join(output_dir, screen_name + ".sh")
            create_shell_script(command, script_filename)
            run_shell_script(script_filename, create_screen, screen_name)

        if create_screen:
            logger.info("done. screen -ls will now show the newly created screens.")


def main():
    parser = ArgumentParser("Create test indices k-fold training and testing. " +
                            "Creates n folds of indices based on the size of the dataset. " +
                            "Indices are stored as pickled numpy arrays in the output folder.\nNB: The optional param" +
                            " --command-line should be something like:\n"
                            "'python myscript.py --output_directory {fold_dir} --myscript_param val " +
                            "--pickled_test_index_file {index_file}'\n",
                            formatter_class=ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')
    parser.add_argument('-o', '--output_directory',
                        required=True,
                        help="directory where logging, backup and output will get stored, must be pickled version")
    parser.add_argument('-n', '--n_folds', type=int,
                        required=True,
                        help='the k in k-folds, the number of folds you want to split data into')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', '--dataset', type=str,
                        default=None,
                        nargs='?',
                        help="dataset to load with pickled_data_loader to determine size of dataset. "
                             "If you know your dataset size, you can use data_size instead.")
    group.add_argument('-s', '--data_size', type=int,
                        default=-1,
                        nargs='?',
                        help='size of the dataset. Used if dataset is not provided')
    parser.add_argument('--command-line', nargs='*', help='creates a shell script for each fold containing this command appended with the output directory and index file')
    syncgp = parser.add_mutually_exclusive_group()
    syncgp.add_argument('--sync', action="store_true", dest="run_synchronized", default=RUN_SYNCHRONIZED,
                        help="if set (the default), run each fold command one after the other in the same screen session (command-line argument must be defined)")
    syncgp.add_argument('--async', action="store_false", dest="run_synchronized",
                        help="if set, run all fold commands at the same time, in seperate screen sessions (command-line argument must be defined)")
    parser.add_argument('--cl-create-screen', action="store_true", default=CREATE_SCREEN, help='if --command-line is set, starts up a gnu screen with the shell script')
    parser.add_argument('--log-level', type=str, default='INFO', nargs='?', help='Output log level [default: %(default)s]')
    params, _ = parser.parse_known_args()
    ready_dir(params.output_directory)
    logging.basicConfig(stream=sys.stderr, level=params.log_level)
    config_file_logging(params.output_directory)
    write_args_file(parser, params.output_directory)
    k_fold_caller(params.output_directory, params.dataset, params.n_folds,
                  params.command_line,
                  data_size=params.data_size,
                  create_screen=params.cl_create_screen,
                  run_synchronized=params.run_synchronized)


if __name__ == '__main__':
    sys.exit(main())
