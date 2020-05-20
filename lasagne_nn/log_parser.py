import os
import ast
import time
import base64
import hashlib
import warnings
from collections import defaultdict
import regex
import pandas as pd
from common.util import find_json_in_string, find_all_in_dict, get_logfile_and_argfile_from_dir, function_name_from_serialized_str
from common.metrics import get_results_from_csv
__author__ = 'Nick'


class LogfileParser(object):
    logfile_regex = dict(
        # key_name=regex to parse value from logfile OR
        # key_name(regex to parse value from logfile, function to convert string to value)
        training_script=r"Logging output of (?P<training_script>.+) run from ",
        train_cl_dir=r"Logging output of .+ run from (?P<train_cl_dir>.+)\n",
        repo_origin=r"\n\tRepo Origin: (?P<repo_origin>.+)\n",
        repo_hash=r"\n\tRepo Hash: (?P<repo_hash>.+)\n",
        machine_platform=r"\n\tPlatform: (?P<machine_platform>.+)\n",
        log_start=(r"\n\tTime: (?P<log_start>.+)\n",
                   lambda log_start: time.strptime(log_start, "%Y%m%d-%H%M%S")),
        arg_file=r"\nwriting args to (?P<arg_file>.+)\n",
        random_seed=(r"\nUsing random seed: (?P<random_seed>\d+\.*\d*)\n", float),
        training_cases=(r"\nNumber of training cases: (?P<training_cases>\d+)\n", int),
        fingerprint_len=(r"\nFingerprint length: (?P<fingerprint_len>\d+)\n", int),
        target_count=(r"\nNumber of targets: (?P<target_count>\d+)\n", int),
        epoch_number=(r"\nEpoch (?P<epoch_number>\d+) took ", int),
        epoch_duration=(r"\nEpoch \d+ took (?P<epoch_time>\d+\.\d+)s\n", float),
        loss_plots=r"\nsaving to (?P<repo_hash>.+)\n",
        weight_file=r"\nsaving network weights to (?P<weight_file>.+)\n",
        loss_type_val=(r"[ ]+(?P<loss_name>.+) (?i)loss:[ ]+(?P<loss_val>\d+\.\d+)\n",
                       lambda loss_name_val: (loss_name_val[0], float(loss_name_val[1]))),
        total_train_time=r"\nTotal training time took (?P<total_train_time>.+) \(h:m:s\)"
    )
    # these keys are created and added to self.parsed_dict after parsing logfile
    logfile_summary_keys = dict(
        # new_key_name=(existing_key_name, function that returns a summarized value when given get_param(existing_key))
        training_script_filename=('training_script', lambda x: os.path.basename(x)),
        training_data_filename=('training_data', lambda x: os.path.basename(x)),
        output_activation_function=('output_activation', function_name_from_serialized_str),
        update_function=('update', lambda x: function_name_from_serialized_str(x['function'])),
        update_args=('update', lambda x: x['kwargs']),
        regularization_function=('penalty', function_name_from_serialized_str),
        regularization_lambda=('lambda', lambda x: None if not isinstance(x, (float, int)) and not x else x),
        average_epoch_duration=('epoch_duration', lambda durations: sum(durations)/float(len(durations))),
        total_trained_epochs=('epoch_number', lambda numbers: sorted(numbers)[-1]),
        num_hidden_layers=('hidden_layers', lambda layers: len(layers)),
        layer_shape=('hidden_layers', lambda layers: [l[0] for l in layers]),
        layer_dropout=('hidden_layers', lambda layers: [l[1] for l in layers]),
        layer_activation=('hidden_layers', lambda layers: [function_name_from_serialized_str(l[2]) for l in layers]),
        multi_instance=('multitask', lambda multitask: multitask),
        single_instance=('multitask', lambda multitask: not multitask),
        sea_stochastic_negs=('negative_blacklist_file', lambda blf: isinstance(blf, basestring) and len(blf) > 0)
    )
    # default values for old logfiles that failed to log specific keys ...
    default_key_values = dict(
        update_function='nesterov_momentum'
    )

    logfile_json_keys = ['training', 'network', 'network_functions']

    def __init__(self, logfile):
        self.logfile = logfile
        with open(self.logfile) as lfs:
            self.log_string = lfs.read()
        self.parsed_dict = self.parse_logfile()
        self.parsed_dict['logfile_summary'] = self.parse_summary_keys()
        self.parsed_dict['logfile_defaults'] = self.get_missing_defaults()

    def get_missing_defaults(self):
        missing_defaults = {}
        for key, default_val in self.default_key_values.items():
            current_key_val = self.get_param(key)
            if (isinstance(current_key_val, (list, tuple)) and not current_key_val) or current_key_val is None:
                missing_defaults[key] = default_val

        # update args
        if not self.get_param('update_args'):
            missing_defaults['update_args'] = dict(
                learning_rate=self.get_param('learning_rate'),
                momentum=self.get_param('momentum')
            )
        return missing_defaults

    def parse_summary_keys(self):
        summary_dict = {}
        for new_key, (existing_key, summarize_func) in self.logfile_summary_keys.items():
            val = self.get_param(existing_key)
            try:
                summary_dict[new_key] = summarize_func(val)
            except BaseException, e:
                warnings.warn('Error while getting summary key {} for existing key {} in {}:\n\t{}'.format(
                    new_key, existing_key, self.logfile, e))
        return summary_dict

    def parse_regex(self):
        parsed_regex = {}
        for key, rgx_and_typecast in self.logfile_regex.items():
            rgx, typ = (rgx_and_typecast, str) if isinstance(rgx_and_typecast, basestring) else tuple(rgx_and_typecast)
            parsed_regex[key] = map(typ, regex.findall(rgx, self.log_string))
            parsed_regex[key] = parsed_regex[key][0] if len(parsed_regex[key]) == 1 else (None if len(parsed_regex[key]) == 0 else parsed_regex[key])
        return parsed_regex

    def parse_json(self):
        json_strings = find_json_in_string(self.log_string.replace('\n', ' ')
                                           .replace('<', "'<").replace('>', ">'")
                                           .replace(" inputs,", " 'inputs',"))
        return dict(zip(self.logfile_json_keys, map(ast.literal_eval, json_strings)))

    def parse_logfile(self):
        parsed_dict = self.parse_regex()
        parsed_dict.update(self.parse_json())
        return parsed_dict

    def get_param(self, *keys):
        v = [find_all_in_dict(k, self.parsed_dict) for k in keys]
        return v[0] if len(keys) == 1 else v

    def get_readable_unique_id(self):
        uniq_keys = ['machine_platform', 'repo_origin', 'repo_hash', 'training_script_filename', 'input_shape',
                     'layer_shape', 'output_shape', 'layer_dropout', 'layer_activation', 'output_activation_function',
                     'objective', 'update_function', 'update_args', 'regularization_function', 'regularization_lambda',
                     'stochastic_negatives', 'negative_threshold', 'positive_negative_ratio', 'sea_stochastic_negs',
                     'multi_instance', 'training_data_filename', 'train_percentage', 'random_seed']
        return '|'.join(map(str, self.get_param(*uniq_keys)))

    def get_unique_hash(self):
        return base64.urlsafe_b64encode(hashlib.md5(self.get_readable_unique_id()).digest())


def get_multiple_network_results(result_files, train_directories):

    keys_pre_update_args, keys_post_update_args = ['input_shape', 'num_hidden_layers', 'layer_shape', 'output_activation_function',
                                 'regularization_function', 'regularization_lambda', 'update_function'], [
                                 'stochastic_negatives', 'negative_threshold', 'positive_negative_ratio', 'sea_stochastic_negs',
                                 'training_data_filename', 'multi_instance',
                                 'training_script_filename', 'training_cases', 'machine_platform',
                                 'random_seed', 'total_trained_epochs', 'average_epoch_duration']
    update_keys = []
    results_df = None
    metric_names = set(['epoch'])
    network_results_dict = defaultdict(list)
    for res_file, network_train_dir in zip(result_files, train_directories):
        network_results_dict[network_train_dir].append(res_file)
    for network_train_dir, network_result_files in network_results_dict.items():
        logfile, _ = get_logfile_and_argfile_from_dir(network_train_dir)
        try:
            # get network info from logfile
            logfile_parser = LogfileParser(logfile)
            network_info_keys = keys_pre_update_args + keys_post_update_args
            network_info_vals = logfile_parser.get_param(*network_info_keys)
            network_info_dict = dict(zip(network_info_keys, network_info_vals))
            update_args = logfile_parser.get_param('update_args')
            for k, v in update_args.items():
                key = 'update_' + k
                network_info_dict[key] = v
                update_keys.append(key)
            network_info_dict['trained_network_hash'] = logfile_parser.get_unique_hash()

            # group all metrics for network by data_set and epoch
            network_result_dict = defaultdict(dict)
            for result_file in network_result_files:
                for metric in get_results_from_csv(result_file):
                    if 'epoch' in metric:
                        network_result_dict[metric['epoch']][metric['label']] = metric['value']
                        network_result_dict[metric['epoch']][metric['label'] + '_result_file'] = result_file
            # compile all results/metrics found for each epoch - dataset pair
            result_list = []
            for epoch, metrics in network_result_dict.items():
                metrics['epoch'] = epoch
                metric_names.update(metrics.keys())
                metrics.update(network_info_dict)
                result_list.append(metrics)
            # create dataframe and append to all network results
            directory_dataframe = pd.DataFrame(result_list)
            if results_df is None:
                results_df = directory_dataframe
            else:
                results_df = results_df.append(directory_dataframe)
        except TypeError as e:
            warnings.warn("couldn't parse logfile {}, skipping result files:\n{}\n({})".format(
                logfile, network_result_files, e))
    first, last = ['epoch'], []
    metric_names = first + [c for c in sorted(metric_names) if c not in first + last] + last
    ordered_columns = keys_pre_update_args + list(set(update_keys)) + keys_post_update_args + ['trained_network_hash'] + metric_names

    if results_df is not None:
        results_df = results_df[list(ordered_columns)]
        results_df['layer_shape'] = map(str, results_df['layer_shape'].values)
    return results_df, ordered_columns, metric_names

