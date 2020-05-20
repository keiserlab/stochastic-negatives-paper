from common.metrics import get_results_from_csv
import re
import pandas as pd


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
    

def get_convergance_by_vote(results_df, metrics, top_n=4):
    converged_fold_dfs = []
    experiment_folds = defaultdict(lambda: defaultdict(Counter))    
    for metric in metrics:
        asc = 'loss' in metric
        for (exp, fold), group in get_best_by_metric(
                results_df, metric, ascending=asc, n=top_n).groupby(['experiment', 'fold']):
            experiment_folds[exp][fold].update(group['epoch'].values.tolist())
    #         print(metric, name, group['epoch'].values)
    for experiment_name, fc in experiment_folds.iteritems():
        for fold, count in fc.iteritems():
            converged_epoch = count.most_common(1)[0][0]
            converged_fold_dfs.append(
                results_df[(results_df['experiment'] == experiment_name) & 
                           (results_df['fold'] == fold) & 
                           (results_df['epoch'] == converged_epoch)]
            ) 
    return pd.concat(converged_fold_dfs)
    
