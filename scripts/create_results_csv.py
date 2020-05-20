import os
import sys
import fnmatch
import pandas as pd
import common.util as ut
import lasagne_nn.log_parser as lp


def find(pattern, path, dironly=False):
    """Find by pattern in path"""
    result = []
    for root, dirs, files in os.walk(os.path.abspath(path)):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                if dironly:
                    result.append(root)
                    break
                else:
                    result.append(os.path.join(root, name))
    return result


def get_log_and_result_files(basepath, pattern_logs='*.log', pattern_results='results.csv', 
                             ignore_result_files_in_train_dir=False):
    """Get trained network logs and result files, optionally ignoring old result files found in the train directory"""
    logfile_dirs = find(pattern_logs, basepath, dironly=True)
    resultfiles = {}
    for logfiledir in logfile_dirs:
        lf, af = ut.get_logfile_and_argfile_from_dir(logfiledir)
        if lf:
            result_files = find(pattern_results, logfiledir)            
            if ignore_result_files_in_train_dir:
                result_files = [rfile for rfile in result_files 
                                if os.path.dirname(rfile) != logfiledir]
            resultfiles[lf] = result_files
            
    return resultfiles


def summarize_stochastic_negative_columns(dataframe, remove_boolean_cols=True):
    """
    Add 'Negative Training Description' and 'sneg_type'. 
    And by default, remove the boolean columns: 'stochastic_negatives', 'sea_stochastic_negs'
    """
    stochastic_negative_columns = ['stochastic_negatives', 'sea_stochastic_negs']
    new_df = pd.Series(
        dataframe[stochastic_negative_columns].sum(axis=1), 
        index=dataframe.index)
    dataframe['sneg_type'] = new_df
    new_df = new_df.replace(2, 'SEA Stochastic Negatives')
    new_df = new_df.replace(1, 'Stochastic Negatives')
    new_df = new_df.replace(0, 'No Stochastic Negatives')
    dataframe['Negative Training Description'] = new_df
    if remove_boolean_cols:
        summary_cols = ['sneg_type', 'Negative Training Description']
        cols_to_delete = ['sneg_type_DELETE', 'Negative Training Description_DELETE']
        dataframe[stochastic_negative_columns[0]] = dataframe[summary_cols[0]]
        dataframe[stochastic_negative_columns[1]] = dataframe[summary_cols[1]]
        dataframe.rename(index=str, columns={summary_cols[0]: cols_to_delete[0]}, inplace=True)
        dataframe.rename(index=str, columns={summary_cols[1]: cols_to_delete[1]}, inplace=True)
        dataframe.rename(index=str, columns={stochastic_negative_columns[0]: summary_cols[0]}, inplace=True)
        dataframe.rename(index=str, columns={stochastic_negative_columns[1]: summary_cols[1]}, inplace=True)
        dataframe.drop(cols_to_delete, axis=1, inplace=True)

    
def summarize_and_order_network_result_df_cols(dataframe, metric_names, columns_to_ignore=None):
    """Clean up network results for csvs by removing unnecessary columns"""
    summarize_stochastic_negative_columns(dataframe)
    columns_to_ignore = columns_to_ignore if columns_to_ignore else []
    # ignore crap metrics
    default_metrics_to_ignore = ['labeled-f2.0--Negative--_drug-matrix', 'binary-auc-t5.0_drug-matrix',
                                 'binary-auc-t6.0_drug-matrix', 'r2_timesplit_validation', 'r2', 'epoch']
    more_metrics_to_ignore = [c for c in metric_names
                              if c.endswith('_result_file') or ('----micro' in c) or ('----macro' in c)]
    metrics_to_ignore = columns_to_ignore + default_metrics_to_ignore + more_metrics_to_ignore
    # order important metrics first
    ordered_metrics = ['binary-auc-t5.0-precision-recall_drug-matrix_result_file', 'r2_train_result_file', 'r2_test', 'r2_train', 'r2_drug-matrix',
                       'r2_drug-matrix-known-values', 'r2_timesplit-validation']
    first_metrics_to_keep = [c for c in ordered_metrics if c in dataframe.columns]
    last_metrics_to_keep = [c for c in metric_names if c not in first_metrics_to_keep + metrics_to_ignore]
    ordered_metrics_to_keep = first_metrics_to_keep + last_metrics_to_keep
    # define indices
    indices_to_ignore = ['average_epoch_duration'] + columns_to_ignore
    indices = [c for c in dataframe.columns if (c not in list(metric_names) + indices_to_ignore)] + ['epoch']
    return dataframe[indices + ordered_metrics_to_keep], indices, ordered_metrics_to_keep
    
    
def get_network_results_in_base_path(basepath, paths_to_ignore=None, ignore_result_files_in_train_dir=False):
    """Get pandas dataframe of results and metadata of all trained networks found under a path in the filesystem"""
    log_and_results = get_log_and_result_files(
        basepath, ignore_result_files_in_train_dir=ignore_result_files_in_train_dir)    
    
    # filter out paths to ignore
    train_logs, result_csvs = [], []
    paths_to_ignore = [os.path.abspath(p) + os.path.sep for p in paths_to_ignore] if paths_to_ignore else []
    for log, results in log_and_results.items():
        if not paths_to_ignore or not any(map(os.path.abspath(log).startswith, paths_to_ignore)):
            # todo: refactor get_multiple_network_results to accept dict
            train_logs += [os.path.dirname(log)] * len(results)
            result_csvs += results

    results_df, ordered_cols, metric_names = lp.get_multiple_network_results(result_csvs, train_logs)
    # TODO: move summarize to lp.get_multiple_network_results ?
    results_df, indexes, metrics = summarize_and_order_network_result_df_cols(results_df, metric_names)
    return results_df, indexes, metrics



if __name__ == '__main__':
    result_csv_file = 'all_nn_results.csv'
    ignore_old_results = True
    result_basepath = '/srv/nas/mk1/users/nmew/nn_results/nn_results/'
    #result_basepath = '/srv/home/nmew/nn_results/stochastic_negatives/multi/7_prcnt_ratio/512_256_128/'
    paths_to_ignore = ['/srv/nas/mk1/users/nmew/nn_results/nn_results/performance_tests',
                       '/srv/nas/mk1/users/nmew/nn_results/nn_results/stochastic_negatives/multi/7_prcnt_ratio/512_1024_2048',
                       '/srv/nas/mk1/users/nmew/nn_results/nn_results/stochastic_negatives/multi/7_prcnt_ratio/1024_2048_3072']

    nn_results_df, inds, mets = get_network_results_in_base_path(result_basepath,
                                                                 paths_to_ignore=paths_to_ignore,
                                                                 ignore_result_files_in_train_dir=ignore_old_results)
    nn_results_df.to_csv(result_csv_file)
    null_results = nn_results_df[nn_results_df['r2_train_result_file'].isnull()]
    missing_r2_test =  null_results['binary-auc-t5.0-precision-recall_drug-matrix_result_file'].unique()
    print(missing_r2_test)
    #print(null_results['epoch'])
    sys.exit(0)


