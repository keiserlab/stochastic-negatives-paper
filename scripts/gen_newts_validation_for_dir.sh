#!/bin/bash
set -x

if [ $# -gt 0 ]
then
  root_dir="$1"
else
  root_dir="$(pwd)"
fi

if [ $# -gt 1 ]
then
  ts_hdf5="$2"
  ts_indx="$3"
else
  ds_dir="/fast/disk0/shared/data/with_PCBA/mid_included/time-split"
  ts_hdf5="${ds_dir}/MI_PCBA_subset_chembl20_800maxmw_10moleculesmin_medianval_mid_4096bvECFP_TS2012val.hdf5"
  ts_indx="${ds_dir}/PCBA_subset_chembl20_800maxmw_10moleculesmin_medianval_mid_target_index_TS2012train.pkl"
fi



script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
repo_dir="$( cd "${script_dir}/.." && pwd )"
validation_script="${repo_dir}/visualizations/r_squared_test_train.py"
default_data_path="/fast/disk0/shared/data"
ignore_data_path="/fast/disk0/shared/data/with_PCBA/mid_included"

parse_from_network_log () {
  train_path="$1"
  key="$2"
  imports="from common.util import get_logfile_and_argfile_from_dir as getlog; from lasagne_nn.log_parser import LogfileParser as lp;"
  echo "$( python -c "${imports} lf, _ = getlog(\"${train_path}\"); print(lp(lf).get_param(\"${key}\"))" )"
}

find_networkscript () {
  # find nn python scripts with names matching a particular directory name of the results
  train_path="$1"
  training_script_filename="$( parse_from_network_log "${train_path}" "training_script_filename" )"
  network_script="${train_path}/${training_script_filename}"
  if [[ ! -e "${network_script}" ]]; then 
    network_script=$( find "${repo_dir}" -name "${training_script_filename}" )
  fi
  if [[ ! -e "${network_script}" ]]; then
    # older network default scripts were renamed, try with newer network prefix
    network_script=$( find "${repo_dir}" -name "lr_nesterov_${training_script_filename}" )
  fi
  echo "${network_script}"
}

find_training_data () {
  # find train hdf5 file used to train this network
  train_path="$1"
  training_dataset="$( parse_from_network_log "${train_path}" "training_data" )"
  if [[ ! -e "${training_dataset}" ]]; then
    training_dataset="$( find ${default_data_path} -name "$( basename ${training_dataset} )" -not -path "${ignore_data_path}/*" )"
  fi
  echo "${training_dataset}"
}

stored_weights_pattern="model_at_epoch_*.npz"
directories=$( find $root_dir -name "$stored_weights_pattern" -printf '%h\n' | sort -u )

for train_path in  $directories
do
  network_script="$( find_networkscript "${train_path}" )"
  training_dataset="$( find_training_data "${train_path}" )"
  training_ds_indx="/srv/nas/mk1/users/ecaceres/20180525_DM_scrubbing/train_data/ts2012_chembl20_MWmax800_scrubDM_minpos10_cutoff5_target_index.pkl"
  if [[ -e "${training_dataset}" ]]
  then

    python ${validation_script} ${train_path} --out_dir "${train_path}/timesplit-validation" --hdf5_file "${ts_hdf5}" --build_nn_script "${network_script}" --test_indices_file '' --set_name 'timesplit-validation' --network_target_map_file "${training_ds_indx}" --dataset_target_map_file "${ts_indx}"

  fi
done

