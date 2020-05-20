__author__ = 'Nick'
import os
import abc
import logging
from functools import partial
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import common.metrics as metrics


def plot_rsquared(prediction, known_value, title="R Squared", img_filename="rsquared_{}.png", result_name='r2',
                  result_suffix=''):
    """
    Plot the r-squared between predicted and known values.
    Args:
        prediction (np.ndarray):
            value to be plotted on y-axis
        known_value (np.ndarray):
            value to be plotted on x-axis
        title (str):
            title of the plot
        img_filename (str):
            format to save as
        result_name (str):
            passed to compute_rsquared
    """
    logging.debug('calc r^2')
    t = known_value.ravel()
    nonzeros = ~np.isnan(t)
    nonzeros[nonzeros] &= t[nonzeros] > 0.0
    p = prediction.ravel()[nonzeros]
    t = t[nonzeros]
    output_dir = os.path.dirname(img_filename) if img_filename is not None else None
    logging.debug('plotting r^2')
    maxv = t.max() + 1
    g = sns.jointplot(x=p, y=t, alpha=0.08, xlim=(0, maxv), ylim=(0, maxv), stat_func=None)
    rsquare = metrics.compute_rsquared(p, t, output_dir=output_dir, result_name=result_name,
                                       result_suffix=result_suffix)
    g = g.annotate((lambda x,y: rsquare), template="{stat}: {val:.3f}",
                   stat="$R^2$", loc="lower right", fontsize=12)
    g.set_axis_labels("Predicted pAC50", "Known pAC50", fontsize=10)
    cmap = plt.cm.get_cmap('BuPu')
    new_color_list = cmap(np.logspace(0.75, 1, 100))
    new_color_list[:, 3] = 0.95
    new_color_list[:-20, 3] = np.linspace(0.05, 0.9, 80)
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n='Al'+cmap.name, a=0, b=100),
        new_color_list)
    g.ax_joint.hexbin(g.x, g.y, mincnt=2, cmap=new_cmap)
    g.fig.tight_layout()
    if img_filename is not None:
        logging.info('saving plot to ' + img_filename)
        g.fig.savefig(img_filename, dpi=150)
        plt.close(g.fig)
    else:
        return g


def plot_labeled_metrics(prediction, truth, label_method=metrics.label_zones, binary_thresh=None,
                         img_filename="labeled_metrics.png", result_suffix='', pred_thresh=None):
    """
    Violin plot showing known labeles on the x axis with the y axis being predicted pAC50 values.
    :param prediction:
    :param truth:
    :param label_method: defaults to metrics.label_zones, must return labeled_y, labeled_x, bins, bin_names
    :param binary_thresh:
    :param img_filename:
    :param result_suffix:
    :return:
    """
    prediction, truth = prediction.ravel(), truth.ravel()
    output_dir = os.path.dirname(img_filename) if img_filename is not None else None
    # convert predictions and truth to multiple labels
    labeled_prediction, labeled_truth, bins, bin_names = label_method(prediction, truth)
    binary = set(bins) == {0, 1}
    # compute labeled metrics
    matthews = metrics.compute_matthews_coefficient(labeled_prediction, labeled_truth,
                                                    output_dir=output_dir, result_suffix=result_suffix)
    f1_each = metrics.compute_labeled_fbeta(labeled_prediction, labeled_truth, beta=1.0, average=None,
                                            label_names=bin_names, output_dir=output_dir, result_suffix=result_suffix)
    f2_each = metrics.compute_labeled_fbeta(labeled_prediction, labeled_truth, beta=2.0, average=None,
                                            label_names=bin_names, output_dir=output_dir, result_suffix=result_suffix)

    hamming = metrics.compute_hamming_loss(labeled_prediction, labeled_truth, bins, result_suffix=result_suffix)
    hamming_each = []
    for b, bn in zip(bins, bin_names):
        inds = np.where(labeled_truth == b)
        hamming_each.append(
            metrics.compute_hamming_loss(labeled_prediction[inds], labeled_truth[inds], bins,
                                         output_dir=output_dir, result_suffix=''.join(('-', bn, result_suffix))))
    # plot violin plot
    maxv = prediction.max() + 1
    fig, g = plt.subplots(figsize=(6,6))
    sns.violinplot(ax=g, x=labeled_truth, y=prediction, scale='area', inner='quartile', ylim=(0, maxv))
    g.set_xticklabels(bin_names)
    
    # overall metrics as lines
    metric_ax = g.twinx()
    f1_color = 'b'
    f2_color = 'c'
    hamming_color = 'y'
    matthews_color = 'g'
    micro_linestyle = 'dotted'
    macro_linestyle = 'dashdot'
    hamming_linestyle = 'dashed'
    line_width = 1.5

    metric_ax.axhline(hamming, linewidth=line_width, color=hamming_color, linestyle=hamming_linestyle, label="hamming loss (overall)")
    metric_ax.axhline(matthews, linewidth=line_width, color=matthews_color, linestyle=hamming_linestyle,
                      label='matthews coeff (overall)')
    # bin metrics as points
    msize = 60
    sctr_styl = dict(alpha=0.75, linewidths=0.8)
    metric_ax.scatter(bins, f1_each, s=msize, c=f1_color, marker='s', label='f1 score', **sctr_styl)
    metric_ax.scatter(bins, f2_each, s=msize, c=f2_color, marker='D', label='f2 score', **sctr_styl)
    metric_ax.scatter(bins, hamming_each, s=msize * 1.5, c=hamming_color, marker='v', label='hamming loss', **sctr_styl)
    metric_ax.set_ylim((np.min([-0.1, matthews - 0.1]) ,1.1))
            
    # display textboxes
    textstr = '\n'.join(("Overall Metrics", "",
                         "hamming loss: \t{:.4f}",
                         "matthews_coeff: \t{:.4f}")).format(hamming, matthews).expandtabs()
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    box_x = 1.18
    box_y = 0.98
    box_y_offset = 0.25
    
    g.text(box_x, box_y, textstr, transform=g.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    
    for i, b in enumerate(bin_names):
        box_y -= box_y_offset
        textstr = '\n'.join(("{} Metrics",
                         "$f1$:\t {:.4f}",
                         "$f2$:\t {:.4f}",
                         "hamming loss: \t{:.4f}")).format(
        b, f1_each[i], f2_each[i], hamming_each[i]).expandtabs()
        g.text(box_x, box_y, textstr, transform=g.transAxes, fontsize=14, verticalalignment='top', bbox=props)
        
    # set label axes
    g.set_xlabel('Known Labels',)
    g.set_ylabel('Predicted pAC50')
    metric_ax.set_ylabel('Metric Score')
    
    # show legends
    metric_ax.legend(bbox_to_anchor=(box_x + 0.7, 1.01), shadow=False, loc="upper left", fontsize=14, fancybox=True)

    # write to file
    if img_filename is not None:
        logging.info('saving plot to ' + img_filename)
        fig.savefig(img_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_tpr_fpr(prediction, truth, known_threshold, thresholds, tpr, fpr, img_filename):
    logging.info('plotting histogram of tpr and fpr')
    fig, ax = plt.subplots()
    sns.kdeplot(truth, shade=True, color=sns.xkcd_rgb["pale green"], ax=ax)
    sns.kdeplot(prediction, shade=True, color=sns.xkcd_rgb["pale orange"], ax=ax)
    label_y = ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0])/5.0
    for t, f, thresh in zip(tpr, fpr, thresholds):
        label = "{:.2f}: tpr={:.2f},  fpr={:.2f}".format(thresh, t, f)
        ax.axvline(thresh, linewidth=1, color=sns.xkcd_rgb["dark grey"], linestyle='dashed', alpha=0.6,
                   label=label)
        ax.annotate(label, xy=(thresh, label_y), xytext=(thresh, label_y), rotation='vertical', size='xx-small')

    if known_threshold is not None:
        ax.axvline(known_threshold, linewidth=0.75, color=sns.xkcd_rgb["pale red"], linestyle='dotted', alpha=0.75,
                   label=label)

    # ax.legend(loc='upper middle', shadow=False)
    if img_filename is not None:
        logging.info('saving plot to ' + img_filename)
        plt.savefig(img_filename, dpi=150)


def plot_binary_auc(prediction, truth, threshold=5.0, fill_truth_nan=4.5, x_y_type='tpr_fpr',
                    ax=None, img_filename=None, result_suffix='', plot_rand=True, show_legend=True, plot_label="", minimal_legend=False, classification=False, **kwargs):
    prediction, truth = prediction.ravel(), truth.flatten()
    unknown = np.isnan(truth)
    truth[np.where(unknown)] = fill_truth_nan
    output_dir = os.path.dirname(img_filename) if img_filename is not None else None
    auc_score, tpr, fpr, thresholds, result_name = metrics.compute_binary_auc(
        prediction, truth, threshold=threshold, x_y_type=x_y_type,
        output_dir=output_dir, result_suffix=result_suffix, classification=classification)
    if ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots()
    if not minimal_legend:
        ax.plot(fpr, tpr, clip_on=False,
             label='{} {} (known-threshold = {:0.2f}, auc = {:0.3f})'.format(plot_label, x_y_type, threshold, auc_score), **kwargs)
    else: 
        ax.plot(fpr, tpr, clip_on=False,
             label='{} (auc = {:0.3f})'.format(plot_label, auc_score), **kwargs)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    if x_y_type == 'tpr-fpr':
        ax.set_title('ROC Curve', fontsize=12)
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        if plot_rand:
            rand_auc_score = 0.5
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, color='0.4')
        if show_legend:
            ax.legend(loc="lower right")
    else:
        if plot_rand:
            binary_truth = truth >= threshold
            rand_auc_score = np.sum(binary_truth)/float(binary_truth.size)
            if not minimal_legend:
                ax.plot([0.0, 1.0], [rand_auc_score, rand_auc_score], 'k--', linewidth=1.5, color='0.4',
                     label='Baseline {} ({:0.3f})'.format(x_y_type, rand_auc_score))
            else:
                ax.plot([0.0, 1.0], [rand_auc_score, rand_auc_score], 'k--', linewidth=1.5, color='0.4',
                     label='Baseline ({:0.3f})'.format(rand_auc_score))
        ax.set_title('Precision Recall Curve', fontsize=12)
        ax.set_ylabel('Precision', fontsize=10)
        ax.set_xlabel('Recall', fontsize=10)
        if show_legend:
            ax.legend(loc="upper right")

    if img_filename is not None:
        logging.info('saving plot to ' + img_filename)
        fig.savefig(img_filename, dpi=150)
        plt.close(fig)
        
    if plot_rand:
        return auc_score, rand_auc_score
    else:
        return auc_score


def plot_multiple_errors(errors_list, error_names, epochs=None, title='Errors', filename=None):
    if epochs is None:
        epochs = np.arange(max(map(len, errors_list)))
    fig, ax = plt.subplots()
    for errors, name in zip(errors_list, error_names):
        ax.plot(epochs, errors, label=name)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    fig.suptitle(title)
    if filename is not None:
        logging.info('saving multiple errors plot to {}'.format(filename))
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_error(errors, epochs=None, title='Errors', filename=None):
    """
    Plot error values over time.
    :param errors:
    :param epochs:
    :param title:
    :param filename:
    :return:
    """
    if epochs is None:
        epochs = np.arange(len(errors))
    plt.plot(epochs, errors)
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.title(title)
    if filename is None:
        filename = 'layerCost.png'
    logging.info('saving to ' + filename)
    plt.savefig(filename)
    plt.clf()


def plot_unknowns(prediction=None, truth=None, unknowns=None, title="Distribution of Unknown Truths",
                  img_filename="unknown_truths_{}.png"):
    """
    Plot histogram of predictions for unknown truths.
    :param prediction: predicted values
    :param truth: known values
    :param unknowns: (optional) unknown values to plot, calculated from prediction and truth if None
    :param title: plot title
    :param img_filename: if not None, image of plot is saved to this path
    :return:
    """
    if unknowns is None:
        logging.info('finding unknowns')
        unknowns = prediction[np.isnan(truth)].ravel()
    if unknowns is None or not len(unknowns):
        return
    logging.info('plotting histogram of {:,} unknowns'.format(len(unknowns)))
    fig, ax = plt.subplots()
    ax.hist(unknowns, bins=50)
    fig.suptitle(title)
    ax.set_title("{:,} datapoints".format(len(unknowns)))
    if img_filename is not None:
        logging.info('saving plot to ' + img_filename)
        fig.savefig(img_filename, dpi=150)
        plt.close(fig)


class MetricsPlot(object):
    # TODO: move calc_and_save_if_necessary functionality here or to a new superclass and remove from viz. functions
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_set_name, stripped_data_set_name=None, **kwargs):
        self.data_set_name = data_set_name
        self.stripped_data_set_name = stripped_data_set_name

    def get_stripped_data_set_name(self):
        return self.stripped_data_set_name or self.data_set_name.lower().replace(',', '').replace(' ', '-')


class UnknownsDistributionPlot(MetricsPlot):
    def __init__(self, data_set_name, epoch, output_dir, unknowns_value=np.nan, stripped_data_set_name=None):
        super(UnknownsDistributionPlot, self).__init__(data_set_name=data_set_name,
                                                       stripped_data_set_name=stripped_data_set_name)
        self.epoch = epoch
        self.output_dir = output_dir
        self.unknowns_value = unknowns_value

    def get_plot_filename(self):
        return os.path.join(self.output_dir, "predictionOfNullsHist_{}_epoch_{}.png".format(
            self.get_stripped_data_set_name(), self.epoch))

    def get_predictions_for_unknowns(self, set_prediction, set_known):
        return set_prediction[set_known == self.unknowns_value]

    def plot(self, set_prediction, set_known, save=True):
        plot_unknowns(set_prediction, set_known,
                      unknowns=self.get_predictions_for_unknowns(set_prediction, set_known),
                      img_filename=self.get_plot_filename() if save else None)


class BinaryLabeledMetricsPlot(MetricsPlot):
    def __init__(self, data_set_name, epoch, output_dir, threshold, stripped_data_set_name=None, pred_thresh=None, regression=True, **kwargs):
        super(BinaryLabeledMetricsPlot, self).__init__(data_set_name=data_set_name,
                                                       stripped_data_set_name=stripped_data_set_name)
        self.epoch = epoch
        self.output_dir = output_dir
        self.threshold = threshold
        self.regression = regression
        
        if not regression:
            if pred_thresh is None:
                # default binary predictor threshold is 0.5
                self.pred_thresh = 0.5
            else:
                self.pred_thresh = pred_thresh
        else:
            if pred_thresh is None:
                # default regressor predictor threshold the same as our true threshold
                self.pred_thresh = threshold
            else:
                self.pred_thresh = pred_thresh

    def get_label_method(self):
        return lambda x, y: metrics.label_binary(x, y, self.threshold, pred_thresh = self.pred_thresh)

    def get_plot_filename(self):
        return os.path.join(self.output_dir, "binary-metrics-true_thresh{}_pred_thresh_{}_{}_epoch_{}.png".format(
            self.threshold, self.pred_thresh, self.get_stripped_data_set_name(), self.epoch))

    def get_result_suffix(self):
        return "_binary-true_thresh{}_pred_thresh_{}_{}_@{}".format(self.threshold, self.pred_thresh, self.get_stripped_data_set_name(), self.epoch)

    def plot(self, set_prediction, set_known, save=True):
        plot_labeled_metrics(set_prediction, set_known,
                             label_method=self.get_label_method(),
                             img_filename=self.get_plot_filename() if save else None,
                             result_suffix=self.get_result_suffix(),
                             binary_thresh=self.threshold)


class RSquaredPlot(MetricsPlot):
    def __init__(self, data_set_name, epoch, output_dir, stripped_data_set_name=None, **kwargs):
        super(RSquaredPlot, self).__init__(data_set_name=data_set_name,
                                           stripped_data_set_name=stripped_data_set_name)
        self.epoch = epoch
        self.output_dir = output_dir

    def get_title(self):
        return "{} R-squared at Epoch {}".format(self.data_set_name, self.epoch)

    def get_plot_filename(self):
        return os.path.join(self.output_dir, "{}_rsquared_epoch_{}.png".format(
            self.get_stripped_data_set_name(), self.epoch))

    def get_result_suffix(self):
        return "_{}_@{}".format(self.get_stripped_data_set_name(), self.epoch)

    def plot(self, set_prediction, set_known, save=True):
        plot_rsquared(set_prediction, set_known,
                      title=self.get_title(),
                      img_filename=self.get_plot_filename() if save else None,
                      result_suffix=self.get_result_suffix())


class AucPlot(MetricsPlot):
    def __init__(self, data_set_name, epoch, threshold, x_y_type, output_dir, stripped_data_set_name=None, **kwargs):
        super(AucPlot, self).__init__(data_set_name=data_set_name,
                                      stripped_data_set_name=stripped_data_set_name)
        self.epoch = epoch
        self.output_dir = output_dir
        self.plot_kwargs = kwargs
        self.threshold = threshold
        self.x_y_type = x_y_type
        self.result_name = metrics.ResultNames.binary_auc

    def get_plot_filename(self):
        return os.path.join(self.output_dir,
                            metrics.get_metric_result_name(
                                self.result_name, self.get_result_suffix(),
                                threshold=self.threshold, type=self.x_y_type)
                            ) + ".png"

    def get_result_suffix(self):
        return "_{}_@{}".format(self.get_stripped_data_set_name(), self.epoch)

    def plot(self, set_prediction, set_known, save=True):
        plot_binary_auc(set_prediction, set_known, x_y_type=self.x_y_type, threshold=self.threshold,
                        img_filename=self.get_plot_filename() if save else None,
                        result_suffix=self.get_result_suffix())

