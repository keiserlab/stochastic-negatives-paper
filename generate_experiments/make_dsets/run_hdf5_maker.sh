#!/bin/sh
# Get classification and regression results.
# Contact: Elena Caceres (ecaceres@keiserlab.org)
# Version: 1

# python script to run
NOTEBOOK_BASE=$HOME"/labgits";
SCRIPT_BASE=$NOTEBOOK_BASE"/lab-notebook-caceres/Projects/nnets/20180525_DM_scrubbing";
make_hdf5=$SCRIPT_BASE"/make_h5file.py";

# bases
EXPT_NAME="20180525_DM_scrubbing";
DATA_DIR=$DATA_SAVE_BASE$EXPT_NAME"/raw_data";
SAVE_DIR=$DATA_SAVE_BASE$EXPT_NAME"/train_data";

# output files
log_file=$SAVE_DIR"/make_hdf5.log";
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
chembl_data=$DATA_DIR"/full_chembl20_cutoff800_dm_scrubbed.csv.gz";
mid2inchi=$DATA_DIR"/mid2inchi.csv.gz";
fp_file=$DATA_DIR"/chembl20_MWmax800_fps.fp.gz";

# output files
output_base_name="chembl20_MWmax800_scrubDM_minpos10_cutoff5"

# activate environment
echo "Activating environment...";
source activate $curr_env;

# run expt
python $make_hdf5 $mid2inchi $SAVE_DIR $fp_file $chembl_data $output_base_name >> $log_file 2>&1;
echo "Finished creating fingerprints";