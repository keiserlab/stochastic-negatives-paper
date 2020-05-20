"""
prediction_analysis.py should handle analysis of network predictions vs known truths
"""
__author__ = 'Nick'
import os
import sys
import glob
import logging
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import common.data_converter as dc
from common.util import config_file_logging, write_args_file, ready_dir
from common.plots import UnknownsDistributionPlot, AucPlot, RSquaredPlot, BinaryLabeledMetricsPlot
import lasagne_nn.output_loader as ol

DM_EXPORT_PATH = '../data/drug_matrix/onlydrugmatrix.csv'


def df_from_chembl_export(export_csv_path, average_duplicates=True, fill_unknowns=True, fill_with=-1, to_pki=True):
    tdf = pd.read_csv(export_csv_path,
                      names=['doc', 'year', 'target', 'compound', 'name', 'affinity', 'relation', 'smiles'],
                      usecols=['target', 'compound', 'affinity'],
                      delimiter='\t',
                      skiprows=[0])
    tdf.set_index(['target', 'compound'], inplace=True)
    tdf.sortlevel(['target', 'compound'], inplace=True)
    # if there are multiple target-compound values, use the mean
    if average_duplicates:
        tdf = pd.DataFrame(tdf.groupby(level=['target', 'compound']).affinity.mean())
    if to_pki:
        tdf.affinity = dc.convert_to_pki(tdf.affinity.values)
    # convert to matrix with nans for missing/unknown values
    tdf = tdf.reset_index().pivot(index='compound', columns='target', values='affinity')
    tdf = pd.DataFrame(tdf.stack(dropna=False), columns=['affinity']).swaplevel(0, 1, axis=0).sort_index()
    if fill_unknowns:
        tdf = tdf.fillna(fill_with)
    return tdf


def intersect_truth_prediction(tdf, pdf):
    logging.debug("pdf.size {}\ttdf.size {}".format(pdf.size, tdf.size))
    logging.debug("pdf.index.names {}\ttdf.index.names{}".format(pdf.index.names, tdf.index.names))
    # drop predictions for which we don't have truths
    pdf = pdf.loc[tdf.index].dropna()
    tdf = tdf.loc[pdf.index]
    logging.debug("pdf.size {}\ttdf.size {}".format(pdf.size, tdf.size))
    logging.debug("pdf.index.names {}\ttdf.index.names{}".format(pdf.index.names, tdf.index.names))
    return tdf, pdf


def plot_drug_matrix_analysis(prediction_path, dm_export_path, override=False, pred_thresh=None, regression=True,):
    tdf, pdf = intersect_truth_prediction(df_from_chembl_export(dm_export_path, fill_unknowns=False),
                                          ol.df_from_prediction_path(prediction_path))
    output_path, output_suffix = os.path.split(prediction_path)
    output_path = ready_dir(output_path)
    output_suffix = os.path.splitext(output_suffix)[0]
    epoch = output_suffix.split("at_epoch_")[-1].split(".")[0].split("_")[0].strip()
    if not epoch or not epoch.isdigit():
        epoch = output_suffix
    
    auc_plot = AucPlot("Drug Matrix", epoch, 5.0, 'tpr-fpr', output_path)
    if not os.path.exists(auc_plot.get_plot_filename()) or override:
        auc_plot.plot(pdf.values, tdf.values)

    auc_plot = AucPlot("Drug Matrix", epoch, 6.0, 'tpr-fpr', output_path)
    if not os.path.exists(auc_plot.get_plot_filename()) or override:
        auc_plot.plot(pdf.values, tdf.values)

    auc_plot = AucPlot("Drug Matrix", epoch, 5.0, 'precision-recall', output_path)
    if not os.path.exists(auc_plot.get_plot_filename()) or override:
        auc_plot.plot(pdf.values, tdf.values)

    auc_plot = AucPlot("Drug Matrix", epoch, 6.0, 'precision-recall', output_path)
    if not os.path.exists(auc_plot.get_plot_filename()) or override:
        auc_plot.plot(pdf.values, tdf.values)

    # If pred_thresh is None and regression = True, then pred_thresh is set to the true_thresh
    # If pred_thresh is None and regression = False, then pred_thresh is set to 0.5
    fbeta_hamming = BinaryLabeledMetricsPlot("Drug Matrix", epoch, output_path, 5.0, 
                                             pred_thresh=pred_thresh, regression=regression)
    if not os.path.exists(fbeta_hamming.get_plot_filename()) or override:
        fbeta_hamming.plot(pdf.values, tdf.fillna(4.5).values)

    fbeta_hamming = BinaryLabeledMetricsPlot("Drug Matrix", epoch, output_path, 6.0, 
                                             pred_thresh=pred_thresh, regression=regression)
    if not os.path.exists(fbeta_hamming.get_plot_filename()) or override:
        fbeta_hamming.plot(pdf.values, tdf.fillna(4.5).values)

    unknown_hist = UnknownsDistributionPlot("Drug Matrix", epoch, output_path)
    if not os.path.exists(unknown_hist.get_plot_filename()) or override:
        unknown_hist.plot(pdf.values, tdf.values)

    all_rsquared_plot = RSquaredPlot("Drug Matrix Unknown Values = 4.5", epoch, output_path)
    if not os.path.exists(all_rsquared_plot.get_plot_filename()) or override:
        all_rsquared_plot.plot(pdf.values, tdf.fillna(4.5).values)

    known_rsquared_plot = RSquaredPlot("Drug Matrix Known Values", epoch, output_path)
    if not os.path.exists(known_rsquared_plot.get_plot_filename()) or override:
        where_not_null = np.where(tdf.notnull().values)
        known_rsquared_plot.plot(pdf.values[where_not_null], tdf.values[where_not_null])


def main():
    parser = ArgumentParser("Prediction Analysis", fromfile_prefix_chars='@')
    parser.add_argument('-p', '--prediction_path', type=str,
                        help="file with chembl compound ids and pki predictions (*.csv)")
    parser.add_argument('--dm_export_path', nargs='?', default=DM_EXPORT_PATH,
                        help="if present, run drug matrix analysis")
    parser.add_argument('-t', '--pred_thresh', type=np.float, default=None,
                        help="Prediction threshold to use. If not set, is set to the true threshold (if regression) or 0.5 (if not regression)")
    parser.add_argument('-r', '--regression', type=bool, default=True,
                        help="Usage: True if regression output, False if classification")
    parser.add_argument('--overwrite', type=bool, default=False, nargs='?',
                        help='True to calulate metrics and plots even if plot files already exist [default: %(default)s]')
    parser.add_argument('--log-level', type=str, default='INFO', nargs='?',
                        help='Output log level [default: %(default)s]')
    params = parser.parse_args()

    logging.basicConfig(stream=sys.stderr, level=params.log_level)
    config_file_logging(os.path.dirname(params.prediction_path))
    write_args_file(parser, os.path.dirname(params.prediction_path))
    for prediction_path in glob.glob(params.prediction_path):
        print("prediction path: {}".format(prediction_path))
        plot_drug_matrix_analysis(prediction_path, params.dm_export_path, override=params.overwrite, pred_thresh=params.pred_thresh, regression=params.regression)


if __name__ == '__main__':
    sys.exit(main())

