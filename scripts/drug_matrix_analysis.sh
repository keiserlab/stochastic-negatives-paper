#!/bin/bash
#Analyze a network against DrugMatrix
#Usage:
USAGE="${BASH_SOURCE[0]} OUTPUT_DIR WEIGHTS_FILE NN_SCRIPT TARGET_FILE [FP_LEN] [PRED_THRESH] [REGRESSION]"
set -x
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NNETBASE="$( cd "${SCRIPT_DIR}/.." && pwd )"
DM_R2_SCRIPT="${NNETBASE}/common/prediction_analysis.py"
RUN_NN_SCRIPT="${NNETBASE}/lasagne_nn/run_nn.py"
DM_SCRPT_PATH="${NNETBASE}/datasets/validation_sets/drug_matrix"
DM_SCRIPT="${DM_SCRPT_PATH}/drug_matix_db_export.sh"
DM_FP_SCRIPT="${DM_SCRPT_PATH}/drug_matrix_fingerprint.sh"
DM_EXPORT="${DM_SCRPT_PATH}/drugmatrix_full_chembl20_cutoff800.csv"
FP_LEN=4096

if [ $# -lt 4 ]
  then
    echo "Usage:\n${USAGE}"
    exit 1
fi

OUTPUT_DIR="$1"
WEIGHTS_FILE="$2"
NN_SCRIPT="$3"
TARGET_FILE="$4"
if [ $# -gt 4 ]
  then
    FP_LEN="$5"
fi
if [ $# -gt 5 ]
  then
    pred_thresh="$6"
    regression="$7"
fi

FP_FILE="${DM_EXPORT%.csv}_${FP_LEN}_fingerprints.fp"
echo ${FP_FILE}

if [[ ! -e "$FP_FILE" ]]
  then 
    echo "generating fingerprints from ${DM_EXPORT}"
    ${DM_FP_SCRIPT} ${FP_LEN} ${FP_FILE} ${DM_EXPORT}
    echo "fingerprints saved to ${FP_FILE}"
  else
    echo "found existing fingerprint file ${FP_FILE}"
fi

source $NNETBASE/set_nnet_path
echo "generating predictions (saving to ${OUTPUT_DIR})"
python $RUN_NN_SCRIPT -o ${OUTPUT_DIR} -t ${TARGET_FILE} -w "${WEIGHTS_FILE}" -n ${NN_SCRIPT} -f ${FP_FILE}
echo -e "done generating predictions (saved to ${OUTPUT_DIR})"

echo "generating plots, saving to ${OUTPUT_DIR}"
if [ $# -gt 5 ]
  then
    python $DM_R2_SCRIPT -p ${OUTPUT_DIR}'/*_prediction.csv' --dm_export_path ${DM_EXPORT} --pred_thresh ${pred_thresh} --regression ${regression}
  else
    python $DM_R2_SCRIPT -p ${OUTPUT_DIR}'/*_prediction.csv' --dm_export_path ${DM_EXPORT}
fi

echo "done generating plots"

#unset nnet_path
unset_nnet

exit 0


