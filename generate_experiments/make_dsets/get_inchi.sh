#!/bin/sh
# Get classification and regression results.
# Contact: Elena Caceres (ecaceres@keiserlab.org)
# Version: 1

# python script to run
NOTEBOOK_BASE=$HOME"/labgits";
SCRIPT_BASE="./generate_experiments/training_data/make_dsets/";
get_inchi=$SCRIPT_BASE"/get_inchi_dict.py";

# bases
EXPT_NAME="20180525_DM_scrubbing";
SAVE_DIR="/srv/nas/mk1/users/ecaceres/"$EXPT_NAME"/raw_data";

# output files
log_file=$SAVE_DIR"/get_inchi.log";
echo "Log file at: $log_file";

# log info
the_notebook="lab-notebook-caceres.wiki/"$EXPT_NAME".md";
right_now=$(date +"%x %r %Z");
curr_env="features"

# make our logfile
echo "Creating log file...";
echo "Expt last run on $HOSTNAME" > $log_file;
echo "Updated on $right_now by $USER" >> $log_file;
echo "For details on this experiment, please see: $the_notebook" >> $log_file;
echo "Anaconda env used: $curr_env" >> $log_file;
echo "Git Info: " >> $log_file;
git log -1 | head -n 2 >> $log_file;

# input files
smiles=$SAVE_DIR"/all_chembl_smiles_mid_mwcutoff800.smi";
inchi_output=$SAVE_DIR"/chembl20_MWmax800_smiles2inchi2mid.csv.gz";

# activate environment
echo "Activating environment...";
source activate $curr_env;

# run expt
time python $get_inchi $smiles $inchi_output >> $log_file 2>&1;
echo "Finished creating inchi";
