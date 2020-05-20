#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"
DMR2_SCRIPT="${SCRIPT_DIR}/drug_matrix_analysis.sh"
HDF5R2_SCRIPT="${SCRIPT_DIR}/gen_validation_for_dir.sh"
STORED_WEIGHTS_PATTERN="model_at_epoch_*.np*"
TARGET_INDEX_FILE_PATTERN="*_target_index*.pkl"
default_data_path="/fast/disk0/shared/data/with_PCBA"
DEFAULT_FP_LEN=4096
OUTPUT_DIR_NAME="drug-matrix"
PROC_PER_GPU=1

skip_dm=false
skip_val=false

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
  if [[ ! -e "${network_script}" ]]
    then 
      network_script=$( find "${REPO_DIR}" -name "${training_script_filename}" )
  fi
  if [[ ! -e "${network_script}" ]]; then
    # older network default scripts were renamed, try with newer network prefix
    network_script=$( find "${REPO_DIR}" -name "lr_nesterov_${training_script_filename}" )
  fi
  echo "${network_script}"
}

find_train_data () {
  train_path="$1"  
  training_dataset="$( parse_from_network_log "${train_path}" "training_data" )"
  if [[ ! -e "${training_dataset}" ]]; then
    training_dataset="$( find "${default_data_path}" -name "$( basename ${training_dataset} )" )"
  fi 
  echo "${training_dataset}"
}


find_target_file () {
  # find train hdf5 file used to train this network
  train_path="$1"
  target_file="$( find "${train_path}" -name "$TARGET_INDEX_FILE_PATTERN")"
  if [[ ! -e "${target_file}" ]]; then
    training_dataset="$( find_train_data "${train_path}" )"
    data_dir="$( dirname "${training_dataset}" )"
    target_file="$( find "${data_dir}" -maxdepth 1 -name "${TARGET_INDEX_FILE_PATTERN}" )"
  fi
  echo "${target_file}"
}

get_fp_len () {
  network_path="$1"
  fp_len="$( parse_from_network_log "${network_path}" "fingerprint_len" )"
  re='^[0-9]+$'
  if ! [[ $fp_len =~ $re ]] ; then
    fp_len="4096"
  fi
  echo ${fp_len}
}


# directories contianing stored weight files
if [ $# -gt 0 ]
then
  root_dir="$1"
else
  root_dir="$(pwd)"
fi

# set prediction and regression thresholds
pred_thresh="$2";
regression="$3";

DIRECTORIES=$( find ${root_dir} -name "$STORED_WEIGHTS_PATTERN" -printf '%h\n' | sort -u )
echo $DIRECTORIES

if [[ "$skip_dm" = false ]]; then
for f in  $DIRECTORIES
do
  # find nn python scripts with names matching a particular directory name of the results
  NETWORKSCRIPT="$( find_networkscript "${f}" )"
  #echo "${NETWORKSCRIPT}"
  if [[ -e "${NETWORKSCRIPT}" ]]
    then
      next_available_gpu=$(/srv/home/nmew/myprojects/neural-nets/scripts/nvidia-available-gpus.sh 1 | head -1)
      while [[ -z $next_available_gpu ]]; do
        echo "no gpus available"
        sleep 90s
        next_available_gpu=$(/srv/home/nmew/myprojects/neural-nets/scripts/nvidia-available-gpus.sh 1 | head -1)
      done


      echo -e "\nfound script for result directory ${f}"
      echo "(${NETWORKSCRIPT})"
      
      alt_replace="${DATA_SAVE_BASE}/20180613_ChEMBL_only_runs/trained_nets"
      OUTPUT_DIR="${f/$TRAINED_NETS_BASE/$default_data_path}"
      OUTPUT_DIR="${OUTPUT_DIR/$alt_replace/$default_data_path}"
      
      OUTPUT_DIR="$OUTPUT_DIR/${OUTPUT_DIR_NAME}"
      mkdir -p $OUTPUT_DIR
      
      FP_LEN="$( get_fp_len "${f}" )"
      TARGET_FILE="$( find_target_file "${f}" )"
      if [[ ! -e "$TARGET_FILE" ]]
        then
          echo "No target file found at ${TARGET_FILE}"
        else      
          DMR2_COMMAND="CUDA_VISIBLE_DEVICES=${next_available_gpu} $DMR2_SCRIPT "${OUTPUT_DIR}" \"${f}/${STORED_WEIGHTS_PATTERN}\" ${NETWORKSCRIPT} ${TARGET_FILE} ${FP_LEN} ${pred_thresh} ${regression}"
          echo "${DMR2_COMMAND}"
          echo "${DMR2_COMMAND}" > "${OUTPUT_DIR}/drugmatrix_analysis.sh"
          eval "${DMR2_COMMAND}" &> "${OUTPUT_DIR}/drugmatrix_analysis.log" &
          PID=$!
          echo "pid: ${PID}"
          echo "log: ${OUTPUT_DIR}/drugmatrix_analysis.log"
          echo -e "\n"
          echo -n "break time "
          naps=5
          napcount=0
          while [[ $napcount -lt $naps ]]; do
            sleep 30s
            echo -n "."
            napcount=$((napcount + 1))
          done
          echo -e "\n\n\n" 
      fi  
    else
      echo "could not find script in ${f}"
  fi
done
echo "waiting on pid: ${PID} ... "
echo "tail log for status: ${OUTPUT_DIR}/drugmatrix_analysis.log"
wait $PID
fi

if [[ "$skip_val" = false ]]; then
for f in  $DIRECTORIES
do
  # find nn python scripts with names matching a particular directory name of the results
  NETWORKSCRIPT="$( find_networkscript "${f}" )"
  #echo "${NETWORKSCRIPT}"
  
  if [[ -e "${NETWORKSCRIPT}" ]]; then
    # next_available_gpu=$(/srv/home/nmew/myprojects/neural-nets/scripts/nvidia-available-gpus.sh 1 | head -1)
    # while [[ -z $next_available_gpu ]]; do
    #   echo "no gpus available"
    #   sleep 30s
    #   next_available_gpu=$(/srv/home/nmew/myprojects/neural-nets/scripts/nvidia-available-gpus.sh 2 | head -1)
    # done
    echo "running $(basename ${HDF5R2_SCRIPT}) in ${f}"
    echo "see log at ${f}/hdf5_analysis.log"
    ${HDF5R2_SCRIPT} ${f} ${pred_thresh} ${regression} &> "${f}/hdf5_analysis.log" &
    echo -e "\n\n"
    echo -n "break time "
    
    naps=5
    napcount=0
    while [[ $napcount -lt $naps ]]; do
      sleep 30s
      echo -n "."
      napcount=$((napcount + 1))
    done
    echo -e "\n\n\n"    
  fi
done
fi
