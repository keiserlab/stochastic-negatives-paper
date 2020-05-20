#!/bin/sh
# Get classification and regression results.
# Contact: Elena Caceres (ecaceres@keiserlab.org)
# Version: 1

# python script to run
NOTEBOOK_BASE=$HOME"/labgits";
SCRIPT_BASE=$NOTEBOOK_BASE"/lab-notebook-caceres/scripts/data_gen";
fp_smiles=$SCRIPT_BASE"/ecfp_from_smiles_mp.py";

# bases
EXPT_NAME="20180525_DM_scrubbing";
SAVE_DIR="/srv/nas/mk1/users/ecaceres/"$EXPT_NAME"/raw_data";

# output files
log_file=$SAVE_DIR"/get_smiles.log";
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
inchi2smiles=$SAVE_DIR"/inchi2smiles.csv.gz";
fp_len="4096";
radius="2";
fp_type="bv";

# output files
out_fps=$SAVE_DIR"/chembl20_MWmax800_fps.fp.gz";


# activate environment
echo "Activating environment...";
source activate $curr_env;

# run expt
python $fp_smiles $inchi2smiles $out_fps -b $fp_len -r $radius -t $fp_type >> $log_file 2>&1;
echo "Finished creating fingerprints";