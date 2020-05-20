#!/bin/bash
# create fingerprints from smiles and predict targets with trained neural network
# Usage (Note the weight file must be in the training output directory): 
# ./script_name.sh outout_dir smiles_file delimiter smiles_column id_column head_rows weight_file
# 
# Example:
# ./predict_from_smile.sh bladder_prediction_1k2k3-arch_4pos-to-3neg ~/tmp/bladder_smiles+fps.csv.gz , 1 0 1 "/fast/disk0/nmew/output/trained_on_full_dataset/four-to-three-pnr_99.9prcnt-train/model_at_epoch_110.npz"

# import functions to find network details from training output dir
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
. ${SCRIPT_DIR}/find_from_train_path.sh

OUTPUT_DIR="$1"
SMILES_FILE="$2"
DELIMIT="$3"
SMI_COL="$4"
CID_COL="$5"
HEAD_ROWS="$6"
WEIGHTS_FILE="$7"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NNETBASE="${SCRIPT_DIR}/.."
RUN_NN_SCRIPT="${NNETBASE}/lasagne_nn/run_nn.py"
ECFP_FROM_SMILES_SCRIPT="${NNETBASE}/datasets/conversion_scripts/ecfp/ecfp_from_smiles_mp.py"



FP_FILE="${OUTPUT_DIR}/fingerprints.csv"

network_path="$(dirname "${WEIGHTS_FILE}" )"
NN_SCRIPT="$( find_networkscript "${network_path}" )"
TARGET_FILE="$( find_target_file "${network_path}" )"

mkdir -p $OUTPUT_DIR

if [ ! -e ${FP_FILE} ]; then
  FP_LEN="$( get_fp_len "${network_path}" )"
  echo "generating fingerprints"
  python $ECFP_FROM_SMILES_SCRIPT $SMILES_FILE $FP_FILE --nBits $FP_LEN --delimiter_in ${DELIMIT} --delimiter_out , --smi_col ${SMI_COL} --mid_col ${CID_COL} --skip_header ${HEAD_ROWS} 
  echo -e "done generating fingerprints\n"
fi

echo "generating predictions (saving to ${OUTPUT_DIR})"
python $RUN_NN_SCRIPT -o ${OUTPUT_DIR} -t ${TARGET_FILE} -w "${WEIGHTS_FILE}" -n ${NN_SCRIPT} -f ${FP_FILE}
echo -e "done generating predictions\n"



