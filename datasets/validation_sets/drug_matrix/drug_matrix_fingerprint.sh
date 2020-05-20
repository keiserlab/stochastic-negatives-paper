#!/bin/bash
# Fingerprint drugmatrix and ready for nn prediction
set -x
set -e 

DM_SCRPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DM_SCRIPT="${DM_SCRPT_PATH}/drug_matrix_db_export.sh"
DM_EXPORT="${DM_SCRPT_PATH}/drugmatrix_full_chembl20_cutoff800.csv"
DM_FILE="${DM_EXPORT}"
ECFP_FROM_SMILES_SCRIPT="${DM_SCRPT_PATH}/../../conversion_scripts/ecfp/ecfp_from_smiles_mp.py"

FP_LEN="$1"
FP_FILE="$2"

if [ $# -gt 2 ]
  then
    DM_FILE="$3"
fi

# export drugmatrix if they want default and it doesn't exist
if [[ ! -e "${DM_FILE}" ]]
  then
    if [[ "$DM_EXPORT" -eq "$DM_FILE" ]]
      then
        $DM_SCRIPT
    fi
fi

# error out if no drugmatrix export
if [[ ! -e "${DM_FILE}" ]]
  then
    echo "no drugmatrix eport found at ${DM_FILE}"
    exit -1
fi

# create fingerprints and format for run_nn
SMILES_FILE="${DM_FILE}_unique.csv"
sort -ru -t$'\t' -k4,4 ${DM_FILE} | tail -n+2 > ${SMILES_FILE}
python $ECFP_FROM_SMILES_SCRIPT $SMILES_FILE $FP_FILE -b $FP_LEN --mid_col 3 --smi_col 7
awk '$1=$1' FS=" " OFS="," ${FP_FILE} > ${FP_FILE}_TMP
mv ${FP_FILE}_TMP ${FP_FILE}
rm ${SMILES_FILE}
exit 0

