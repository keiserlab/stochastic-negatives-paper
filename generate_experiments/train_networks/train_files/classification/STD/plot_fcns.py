import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_env_var(handle):
    tmp = os.getenv(handle)
    if not tmp:
        raise LookupError("Environment variable: {} not set.".format(handle))
    return tmp.strip("'")

def get_metrics(df, list_of_strs):
    return df[list_of_strs]

def get_str_metrics(df, col_name, list_of_strs):
    return df[df[col_name].isin(list_of_strs)]


def format_for_sbs_plotting(df, list_of_metrics):
    first = True
    for i in list_of_metrics:
        tmp_df = df.loc[:,[i, "SN_type", "fold", "index"]]
        tmp_df.columns = ["Value", "SN_type", "fold", "index"]
        tmp_df["Metric"] = i
        if first: 
            new_df = tmp_df
            first = False
        else: 
            new_df = pd.concat([new_df, tmp_df])
    return new_df
    
def group_to_compare(summary_df, reference, metric, group=["index"],):
    ref = get_str_metrics(summary_df, "SN_type", [reference]).groupby(group).mean()
    d_sets = summary_df.SN_type.unique().tolist()
    first = True
    for d in d_sets:
        expt_ref = get_str_metrics(summary_df, "SN_type", [d]).groupby(group).mean()
        tmp_df = ref.subtract(expt_ref)[[metric]]
        tmp_df["SN_type"] = d
        if first:
            df = tmp_df
            first = False
        else:
            df = pd.concat([df, tmp_df])
    return df


def data_to_plot(summary_df, compare_to, metric, mean_tp_fps, plot_this):
    df = group_to_compare(summary_df, compare_to, plot_this)
    new_df = pd.concat([df, mean_tp_fps], axis=1, join_axes=[df.index])
    return new_df


def plot(df, compare_to, metric, eval_dset, top=0.9, order=2, hline_width=1.5, hline_alpha=0.5,
         pt_size=2, pt_alpha=0.7,**kwargs):
    # get the data & consolidate
    sns.despine(bottom=True, left=True)
    sns.set_context("poster")
    n_groups = len(df.SN_type.unique().tolist())
    kwargs["palette"] = sns.color_palette("Set2", n_groups)
    
    # plot data
    g = sns.FacetGrid(df, **kwargs) 
    g.map(sns.regplot, "neg_percentage", metric, order=order, scatter_kws={"s":pt_size, "alpha":pt_alpha})
    g.fig.subplots_adjust(top=top)
    g.fig.suptitle("{} targetwise difference to {} ({})".format(metric, compare_to, eval_dset))
    for a in g.axes:
        a.axhline(0.0, alpha=hline_alpha, color='grey', linestyle="--", linewidth=hline_width)