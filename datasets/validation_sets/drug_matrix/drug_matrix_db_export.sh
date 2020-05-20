#!/bin/bash
# Export DrugMatrix from chembl20 using datasets/db_chembl20_queries/mysql_generate_raw_data.py 
set -e
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CHEMBL20_DB_SCRIPT="${SCRIPTDIR}/../../db_chembl20_queries/mysql_generate_raw_data.py"
OUTFILE="${SCRIPTDIR}/drugmatrix_full_chembl20_cutoff800.csv"
if [ $# -lt 1 ]
  then
    echo "please provide chembl database username"
    exit -1
  else
    UNAME="$1"
fi

python ${CHEMBL20_DB_SCRIPT} -d -a -c 800 ${UNAME} ${OUTFILE}

echo "drugmatrix exported to ${OUTFILE}"

