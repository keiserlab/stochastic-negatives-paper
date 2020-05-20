#!/bin/bash
set -v

sea_blacklist_script="/srv/home/nmew/myprojects/neural-nets/datasets/train_sets/stochastic_negative_blacklists/generate_sea_predicted_blacklist.py"
target_index_file="/srv/nas/mk1/users/ecaceres/20180525_DM_scrubbing/train_data/ts2012_chembl20_MWmax800_scrubDM_minpos10_cutoff5_target_index.pkl"
output_dir=$HOME_SAVE_BASE"/output/20190410_SMA_Investigation/SEA_GEN/binding/preds"
sea_prediction_file="${output_dir}/train_ts2012_chembl20_MWmax800_scrubDM_minpos10_cutoff5_SEA_predictions.csv"
output_file="${output_dir}/MI_PCBA_subset_chembl20_800maxmw_10moleculesmin_medianval_mid_4096bvECFP_TS2012train_sea_pcutoffe-5_blacklist.csv.gz"
cutoff="1.0e-5"

python $sea_blacklist_script $target_index_file $sea_prediction_file $output_file $cutoff

