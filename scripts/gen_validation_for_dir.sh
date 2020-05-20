#!/bin/bash
# makes the train/test/validation predictions and analyses. Does NOT do Drug Matrix
set -x

if [ $# -gt 0 ]
then
  root_dir="$1"
else
  root_dir="$(pwd)"
fi

# set prediction and regression thresholds
pred_thresh="$2";
regression="$3";
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
repo_dir="$( cd "${script_dir}/.." && pwd )"
validation_script="${repo_dir}/visualizations/r_squared_test_train.py"
default_data_path="/srv/nas/mk1/users/ecaceres/nnet_output/"

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
    training_dataset="$( find ${default_data_path} -name "$( basename ${training_dataset} )" )"
  fi
  echo "${training_dataset}"
}

stored_weights_pattern="model_at_epoch_*.npz"
directories=$( find -L $root_dir -name "$stored_weights_pattern" -printf '%h\n' | sort -u )


# load gpu availablity functions
# . /srv/home/ecaceres/nvidia-available-gpus.sh 3;

for train_path in $directories
do
  network_script="$( find_networkscript "${train_path}" )"
  training_dataset="$( find_training_data "${train_path}" )"
  
  alt_replace="${DATA_SAVE_BASE}/20180613_ChEMBL_only_runs/trained_nets"
  out_dir="${train_path/$alt_replace/$default_data_path}"
  out_dir="${out_dir/$TRAINED_NETS_BASE/$default_data_path}"
  
  mkdir -p $out_dir
  
  if [[ -e "${training_dataset}" ]]
  then
    next_available_gpu=$(/srv/home/nmew/myprojects/neural-nets/scripts/nvidia-available-gpus.sh 1 | head -1)
    while [[ -z $next_available_gpu ]]; do
      echo -n "."
      sleep 60s
      next_available_gpu=$(/srv/home/nmew/myprojects/neural-nets/scripts/nvidia-available-gpus.sh 2 | head -1)
    done
    
    RUN_TRAIN="CUDA_VISIBLE_DEVICES=${next_available_gpu} python ${validation_script} ${train_path} --out_dir ${out_dir}/test-train --hdf5_file ${training_dataset} --build_nn_script ${network_script} --pred_thresh ${pred_thresh} --regression ${regression}"
    echo "${RUN_TRAIN}"
    eval "${RUN_TRAIN}"
  
    train_substr="train_ts";
    val_substr="val_ts";
    echo "train set found"
    
    echo "${network_script}"
    echo "${training_dataset_script}"
    validation_dataset="${training_dataset/$train_substr/$val_substr}"
    
    if [[ -e ${validation_dataset} ]]; then
      next_available_gpu=$(/srv/home/nmew/myprojects/neural-nets/scripts/nvidia-available-gpus.sh 1 | head -1)
      while [[ -z $next_available_gpu ]]; do
        echo -n "."
        sleep 60s
        next_available_gpu=$(/srv/home/nmew/myprojects/neural-nets/scripts/nvidia-available-gpus.sh 2 | head -1)
      done
      
      RUN_TEST="CUDA_VISIBLE_DEVICES=${next_available_gpu} python ${validation_script} ${train_path} --out_dir ${out_dir}/timesplit-validation --set_name timesplit-validation --hdf5_file ${validation_dataset} --test_indices_file '' --build_nn_script ${network_script} --pred_thresh ${pred_thresh} --regression ${regression}"
      echo "${RUN_TEST}"
      eval "${RUN_TEST}"
      echo "validation set found"
    else
      validation_dataset="${DATA_SAVE_BASE}/20180525_DM_scrubbing/train_data/val_ts2012_chembl20_MWmax800_scrubDM_minpos10_cutoff5.hdf5"
      if [[ -e ${validation_dataset} ]]; then
        next_available_gpu=$(/srv/home/nmew/myprojects/neural-nets/scripts/nvidia-available-gpus.sh 1 | head -1)
        while [[ -z $next_available_gpu ]]; do
          echo -n "."
          sleep 60s
          next_available_gpu=$(/srv/home/nmew/myprojects/neural-nets/scripts/nvidia-available-gpus.sh 2 | head -1)
        done
        RUN_TEST="CUDA_VISIBLE_DEVICES=${next_available_gpu} python ${validation_script} ${train_path} --out_dir ${out_dir}/timesplit-validation --set_name timesplit-validation --hdf5_file ${validation_dataset} --test_indices_file '' --build_nn_script ${network_script} --pred_thresh ${pred_thresh} --regression ${regression}"
        echo "${RUN_TEST}"
        eval "${RUN_TEST}"
        echo "validation set found"
      else
        echo "no validation script found for network: ${train_path}"
      fi
    fi
  else
    echo "no training script found for network: ${train_path}"
  fi
done

set +x