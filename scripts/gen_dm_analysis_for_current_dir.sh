#!/bin/bash
set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"
DMR2_SCRIPT="${SCRIPT_DIR}/drug_matrix_analysis.sh"
STORED_WEIGHTS_PATTERN="model_at_epoch_*.np*"
TARGET_INDEX_FILE_PATTERN="*_target_index*.pkl"
default_data_path="/fast/disk0/shared/data/with_PCBA"
DEFAULT_FP_LEN=4096
OUTPUT_DIR_NAME="drug-matrix"
if [ $# -gt 0 ]
then
  RESULT_DIR="$1"
else
  RESULT_DIR="$(pwd)"
fi


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


#echo " find $RESULT_DIR -name \"$STORED_WEIGHTS_PATTERN\" -printf '%h\n' | sort -u"
DIRECTORIES=$( find $RESULT_DIR -name "$STORED_WEIGHTS_PATTERN" -printf '%h\n' | sort -u )
echo ${DIRECTORIES}

for f in  $DIRECTORIES
do
  # find nn python scripts with names matching a particular directory name of the results
  NETWORKSCRIPT="$( find_networkscript "${f}" )"
  echo -e "${f}";
  echo -e "${NETWORKSCRIPT}"
  if [[ -e "${NETWORKSCRIPT}" ]]
    then
      echo -e "\nfound script for result directory $f"
      echo "(${NETWORKSCRIPT})"
      OUTPUT_DIR="${f}/${OUTPUT_DIR_NAME}"
      mkdir -p "${OUTPUT_DIR}"
      FP_LEN="$( get_fp_len "${f}" )"
      TARGET_FILE="$( find_target_file "${f}" )"
      if [[ ! -e "$TARGET_FILE" ]]
        then
          echo "No target file found at ${TARGET_FILE}"
        else 
          # rm ${OUTPUT_DIR}/model_at_epoch_*_prediction.csv
          DMR2_COMMAND="$DMR2_SCRIPT "${OUTPUT_DIR}" \"${f}/${STORED_WEIGHTS_PATTERN}\" ${NETWORKSCRIPT} ${TARGET_FILE} ${DM_EXPORT_PATH} ${FP_LEN}"
          echo "${DMR2_COMMAND}"
          echo "${DMR2_COMMAND}" > "${OUTPUT_DIR}/drugmatrix_analysis.sh"
          eval "${DMR2_COMMAND}" &> "${OUTPUT_DIR}/drugmatrix_analysis.log" &
          PID=$!
          echo "pid: ${PID}"
          echo "log: ${OUTPUT_DIR}/drugmatrix_analysis.log"
      fi  
    else
      echo "could not find script in ${f}"
  fi
done


