__author__ = 'Nick'
import os
import json
import warnings
from string import Formatter
import numpy as np
from sklearn.metrics import fbeta_score, hamming_loss, matthews_corrcoef, auc, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from scipy.stats import linregress

RESULTS_FILE = 'results.csv'
JSON_RESULTS_FILE = 'results.json'


class ResultNames(object):
    binary_auc = 'binary-auc-t{threshold}-{type}'
    f_beta = 'labeled-f{beta}--{label}--{average}'
    hamming_loss = 'hamming-loss'
    accuracy = 'accuracy'
    matthews_corrcoef = 'matthews-corrcoef'
    r_squared = 'r2'

    @staticmethod
    def iter_names():
        for metric, name in ResultNames.__dict__.items():
            if not metric.startswith('__') and isinstance(name, basestring):
                yield name


def parse_metric_from_results_label(label):
    label_format = None
    epoch = None
    label = label.replace('_model_at_epoch_', '@').replace('_prediction', '')  # handle old result labels

    if '@' in label:
        label, epoch = label.split('@')
        epoch = int(epoch)

    # handle old labels
    label = label.replace('drug-matrix-unknown-values-=-4.5', 'drug-matrix')
    if 'r2_with_unknown' in label:
        label = 'r2_drug-matrix'
    elif 'r2_known' in label:
        label = 'r2_drug-matrix-known-values'
    elif label.startswith('val_r2'):
        label = 'r2_timesplit-validation'

    for result_name in ResultNames.iter_names():
        if label.startswith(Formatter().parse(result_name).next()[0]):
            label_format = result_name
            break
    if label_format is not None:
        # handle recent results
        data_set = label.rstrip('_').split('_')[-1]
        metric_name = label.split('_', 1)[0]
    else:
        # if it didn't match existing labels, assume old r2 label
        metric_name = 'r2'
        if 'train' in label:
            label = 'r2_train'
            data_set = 'train'
        elif 'test' in label:
            label = 'r2_test'
            data_set = 'test'
        else:
            data_set = 'all/unknown'

    metric = dict(
        name=metric_name,
        # meta=meta,
        data_set=data_set,
        label=label.rstrip('_'),
        label_format=label_format
    )
    if epoch is not None:
        metric['epoch'] = epoch
    return metric


def get_results_from_csv(result_file):
    results = []
    with open(result_file, 'r') as rf:
        for line_number, line in enumerate(rf):
            splitline = line.split(',')
            if len(splitline) == 2:
                label, value = line.split(',')
                try:
                    metric = parse_metric_from_results_label(label)
                    metric['value'] = float(value)
                    if 'epoch' not in metric:
                        warnings.warn('no epoch on line {} of {}, using line number instead\n\t\t{}'.format(
                            line_number, result_file, line))
                        metric['epoch'] = line_number
                    results.append(metric)
                except ValueError:
                    warnings.warn("ValueError while parsing result file {}.\nline:{}\nlabel:{}".format(
                        result_file, line, label))
            else:
                if line.strip() != '':
                    warnings.warn("Too many results on one line ({}): {} ".format(len(splitline), result_file))
    return results


def output_json_results(output_path, result_dict, result_filename=JSON_RESULTS_FILE):
    """
    writes name and value to a csv in the given output path. Creates new csv if none exists.
    :param output_path: directory in which to write results json
    :param result_dict: dict with result value, and metadata such as epoch, dataset, etc. to be written as json
    :return:
    """
    with open(os.path.join(output_path, result_filename), 'a') as rf:
        rf.write(json.dumps(result_dict))


def output_results(output_path, result_name, result_value, result_filename=RESULTS_FILE):
    """
    writes name and value to a csv in the given output path. Creates new csv if none exists.
    :param output_path: directory in which to write results csv
    :param result_name: label written to csv
    :param result_value: value written to csv
    :return:
    """
    with open(os.path.join(output_path, result_filename), 'a') as rf:
        rf.write('{},{}\n'.format(result_name, result_value))


def label_binary(prediction, truth, threshold, pred_thresh=None):
    """
    Returns values formatted for use with various scipy binary reporting metrics
    :param prediction: predictions from model
    :param truth: ground truth value
    :param threshold: threshold for the true and predicted values
    :param pred_thresh: np.float if not None, threshold to use on predicted values (e.g. 0.5 for binary)
    :return: predicted values >= threshold (or pred_thresh if given), truth masked for NaNs and greater than threshold, [0,1] for formatting, [Negative, Positive] (what 0,1 stand for in prev return).
    """
    # prediction is gteq threshold, truth is gteq threshold and not nan
    truth_gteq = ~np.isnan(truth)
    mask = truth_gteq.copy()
    
    # gives values that 1) aren't NaN and 2) have vals > thresh
    truth_gteq[truth_gteq] &= truth[truth_gteq] >= threshold
    
    # This allows us to use a custom threshold if you'd prefer to use a 
    # different thresh for regression or use a binary predictor
    if pred_thresh is not None:
        # if the custom prediction is set
        pred_gteq = ~np.isnan(prediction)
        pred_gteq[mask] = prediction[mask] >= pred_thresh
    else:
        # else, use same as truth thresh
        pred_gteq = ~np.isnan(prediction)
        pred_gteq[mask] = prediction[mask] >= threshold
        
    return pred_gteq, truth_gteq, [0, 1], ['Negative', 'Positive']


def label_bins(prediction, truth, bin_width=2.0):
    """
    puts predictions and truths into bins between -1 and max(truth) + bin_width. Unknowns are set to -1
    :param prediction:
    :param truth:
    :param bin_width:
    :return:
    """
    knowns = np.where(~np.isnan(truth))
    truth_labels = np.zeros_like(truth) - 1
    truth_labels[knowns] = bin_width * np.round(truth[knowns] / bin_width)
    prediction_labels = np.zeros_like(truth) - 1
    prediction_labels = bin_width * np.round(prediction / bin_width)
    bins = [-1] + list(np.arange(0, bin_width + bin_width * round(np.max(prediction) / bin_width), bin_width))
    bin_names = ['PN']
    for i, b in enumerate(bins[1:], 1):
        bin_names.append(str(b))

    return prediction_labels, truth_labels, bins, bin_names


def label_zones(prediction, truth, zone_boundaries=(5.5, 7, 9, 12)):
    """
    puts predictions and truths into zones between given boundries. see numpy.digitize
    :param prediction:
    :param truth:
    :param zone_boundaries: Array of zone boundries. It has to be 1-dimensional and monotonic.
    :return:
    """
    truth_labels = np.array(truth)
    truth_labels[np.isnan(truth_labels)] = -np.inf
    bins = range(0, len(zone_boundaries) + 1)
    prediction_labels = np.digitize(prediction, zone_boundaries)
    truth_labels = np.digitize(truth_labels, zone_boundaries)
    bin_names = ['pAC50 < {}'.format(zone_boundaries[0])]
    for i, b in enumerate(zone_boundaries[1:], 1):
        bin_names.append('{}-{} pAC50'.format(zone_boundaries[i-1], zone_boundaries[i]))
    bin_names.append('pAC50 > {}'.format(zone_boundaries[-1]))
    return prediction_labels, truth_labels, bins, bin_names


def binary_tpr_fpr(labeled_prediction, labeled_truth):
    """
    https://en.wikipedia.org/wiki/Sensitivity_and_specificity
    tpr = sum(pp & kp)/sum(kp)
    fpr = sum(pp & kn)/sum(kn)
    :param labeled_prediction:
    :param labeled_truth:
    :return:
    """
    negative = ~labeled_truth
    tpr = np.sum(labeled_truth & labeled_prediction) / np.sum(labeled_truth, dtype=np.float)
    fpr = np.sum(negative & labeled_prediction) / np.sum(negative, dtype=np.float)
    return tpr, fpr


def binary_precision_recall(labeled_prediction, labeled_truth):
    """
    https://en.wikipedia.org/wiki/Precision_and_recall
    precision                  = tp/tp+fp = tp/pp = sum(pp & kP)/sum(pp)
    recall = tpr = sensitivity = tp/tp+fn = tp/kp = sum(pp & kp)/sum(kp)
    :param labeled_prediction:
    :param labeled_truth:
    :return:
    """
    tp = np.sum(labeled_truth & labeled_prediction)
    precision = tp / np.sum(labeled_prediction, dtype=np.float)
    precision = 0.0 if np.isnan(precision) else precision
    recall = tp / np.sum(labeled_truth, dtype=np.float)
    recall= 1.0 if np.isnan(recall) else recall
    return precision, recall


def get_metric_result_name(metric_result_name, result_suffix, **kwargs):
    return metric_result_name.format(**kwargs) + result_suffix


def compute_binary_auc(prediction, truth, threshold=5.0, x_y_type='tpr-fpr', output_dir=None,
                       result_name=ResultNames.binary_auc, result_suffix='', classification=False):
    """

    Computes auc using sklearn (http://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html) by
    splitting into positives and negatives at various thresholds and computing fpr and tpr or precision for each.
    Writes result_name and hamming loss to results.csv in output_dir if output_dir is not None.
    :param prediction:array_like
    :param truth:array_like
    :param x_y_type: str can be 'tpr-fpr' or 'precision-recall'
    :param output_dir:str if not None, uses output_results to write result to file
    :param result_name:str if output_dir is not None, uses this as label for result
    :param result_suffix: appended to result_name
    :return: auc_score, tpr, fpr, thresholds
    """
    
    if classification: 
        binary_truth = truth >= threshold
        if x_y_type == "tpr-fpr":
            # x is fpr, y is tpr
            x, y, thresholds = roc_curve(binary_truth, prediction)
            auc_score = roc_auc_score(binary_truth, prediction)
        else:
            x_y_type = 'precision-recall'
            # y is precision, x is recall
            y, x, thresholds = precision_recall_curve(binary_truth, prediction)
            auc_score = average_precision_score(binary_truth, prediction)
        formatted_result_name = get_metric_result_name(result_name, result_suffix, threshold=threshold, type=x_y_type)
    else:
        # Compute ROC curve and ROC area for each class
        step = 0.05
        start = np.min(prediction)
        stop = np.max(prediction)
        thresholds = np.arange(start, stop, step)
        x = np.zeros(len(thresholds))
        y = np.zeros(len(thresholds))
        binary_truth = truth >= threshold
        if x_y_type == 'tpr-fpr':
            binary_x_y = binary_tpr_fpr
            x = np.append(x, [1.0, 0.0])
            y = np.append(y, [1.0, 0.0])
        else:
            x_y_type = 'precision-recall'
            binary_x_y = binary_precision_recall

        for i, t in enumerate(thresholds):
            binary_prediction = prediction >= t
            y[i], x[i] = binary_x_y(binary_prediction, binary_truth)

        if x_y_type == 'precision-recall':
            # append start and end (x starts at zero and y ends at baseline P/N+P)
            arg_min = np.nanargmin(x)
            arg_min = arg_min[0] if isinstance(arg_min, np.ndarray) else arg_min
            x = np.append(x, [0.0, 1.0])
            y = np.append(y, [y[arg_min], np.sum(binary_truth)/float(binary_truth.size)])

        # adjust thresholds for appended start, end points
        thresholds = np.append(thresholds, [start - step, stop + step])
        # compute area under curve
        order = np.lexsort((y, x))
        y, x, thresholds = y[order], x[order], thresholds[order]
        auc_score = auc(x, y)
    
    formatted_result_name = get_metric_result_name(result_name, result_suffix, threshold=threshold, type=x_y_type)
    if output_dir is not None:
        output_results(output_dir, formatted_result_name, auc_score)
        
    return auc_score, y, x, thresholds, formatted_result_name


def compute_matthews_coefficient(labeled_prediction, labeled_truth, output_dir=None,
                                 result_name=ResultNames.matthews_corrcoef, result_suffix=''):
    """
    Computes Matthews correlation coefficient using sklearn
    (http://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)
    Writes result_name and hamming loss to results.csv in output_dir if output_dir is not None.
    :param labeled_prediction: Array with predicted label at each position
    :param labeled_truth: Array with true/known label at each position
    :param output_dir:str if not None, uses output_results to write result to file
    :param result_name:str if output_dir is not None, uses this as label for result
    :param result_suffix: appended to result_name
    :return:
    """
    corr = matthews_corrcoef(labeled_prediction, labeled_truth)
    
    if output_dir is not None:
        output_results(output_dir, result_name + result_suffix, corr)
    return corr


def compute_hamming_loss(labeled_prediction, labeled_truth, bins=None, output_dir=None,
                         result_name=ResultNames.hamming_loss, result_suffix=''):
    """
    Computes hamming loss using sklearn
    (http://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html)
    Writes result_name and hamming loss to results.csv in output_dir if output_dir is not None.
    :param labeled_prediction: Array with predicted label at each position
    :param labeled_truth: Array with true/known label at each position
    :param bins:
    :param output_dir:str if not None, uses output_results to write result to file
    :param result_name:str if output_dir is not None, uses this as label for result
    :param result_suffix: appended to result_name
    :return:
    """
    loss = hamming_loss(labeled_prediction, labeled_truth, bins)
    if output_dir is not None:
        output_results(output_dir, result_name + result_suffix, loss)
    return loss


def compute_labeled_fbeta(labeled_prediction, labeled_truth, beta=1., labels=None, average=None, label_names=None,
                          output_dir=None, result_name=ResultNames.f_beta, result_suffix=''):
    """
    Computes f-score using sklearn (http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html).
    Writes result_name and fbeta score(s) to results.csv in output_dir if output_dir is not None.
    :param labeled_prediction: Array with predicted label at each position
    :param labeled_truth: Array with true/known label at each position
    :param labels: (optional) see http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html
    :param average: defaults None see http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html
    :param output_dir:str if not None, uses output_results to write result to file
    :param result_name:str if output_dir is not None, uses this as label for result
    :param result_suffix: appended to result_name
    :return:
    """
    if average == "micro":
        loss = fbeta_score(labeled_prediction, labeled_truth, beta=beta, average=average)
    else: 
        loss = fbeta_score(labeled_prediction, labeled_truth, beta=beta, labels=labels, average=average)
    
    if output_dir is not None:
        result_name += result_suffix
        average = '' if average is None else average
        if type(loss) is float or type(loss) is np.float64:
            label = ''
            result_format_dict = dict([(key, val) for key, val in (
                ('beta', beta), ('label', label), ('average', average)) if key.join('{}') in result_name])
            result_name = result_name.format(**result_format_dict)
            output_results(output_dir, result_name, loss)
        else:
            labels = labels or range(0, len(loss))
            label_names = label_names or labels
            for label, name, label_loss in zip(labels, label_names, loss):
                result_format_dict = dict([(key, val) for key, val in (
                    ('beta', beta), ('label', name), ('average', average)) if key.join('{}') in result_name])
                filled_result_name = result_name.format(**result_format_dict)
                output_results(output_dir, filled_result_name, label_loss)
    return loss


def compute_rsquared(prediction, truth, output_dir=None, result_name=ResultNames.r_squared, result_suffix=''):
    """
    Computes r**2 value of two numpy arrays.
    :param prediction:
    :param truth:
    :param output_dir: if not None, uses output_results to write result to file
    :param result_name: if output_dir is not None, uses this as label for result
    :param result_suffix: appended to result_name
    :return:
    """
    slope, intercept, r_value, p_value, std_err = linregress(truth, prediction)
    r2 = r_value ** 2
    if output_dir is not None:
        output_results(output_dir, result_name + result_suffix, r2)
    return r2
