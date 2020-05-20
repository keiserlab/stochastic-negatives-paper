#!/bin/sh
# Generate k-folds for training
# Contact: Elena Caceres (ecaceres@keiserlab.org)
# Version: 1

# input files
expt_base=$DATA_SAVE_BASE"/20180525_DM_scrubbing/train_data";
dataset=$expt_base"/train_ts2012_chembl20_MWmax800_scrubDM_minpos10_cutoff5.hdf5";
SAVE_DIR=$expt_base"/kfold_indices";
folds="5"

# python script to run
NOTEBOOK_BASE=$HOME"";
SCRIPT_BASE=$HOME"/labgits/neural-nets/common/";
gen_kfolds=$SCRIPT_BASE"/kfold_cl.py";

# bases
EXPT_NAME="20180613_ChEMBL_only_runs";

# output files
log_file=$SAVE_DIR"/get_kfold_indices.log";
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


# activate environment
echo "Activating environment...";
source activate $curr_env;

# make directory & parents if not exists
mkdir -p $SAVE_DIR

# run expt
python $gen_kfolds -o $SAVE_DIR -d $dataset -n $folds >> $log_file 2>&1;
echo "Finished creating k-folds";