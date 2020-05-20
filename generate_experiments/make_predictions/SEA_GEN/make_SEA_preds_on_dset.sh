#!/bin/sh
# Get targetwise performance for nnet predictions
# Contact: Elena Caceres (ecaceres@keiserlab.org)
# Version: 1
# python script to run

make_predictions=$HOME"/labgits/lab-notebook-caceres/scripts/prediction_handling/hdf5_sea_prediction_script.py";

# inputs
hdf5_file=$DATA_SAVE_BASE"/20180525_DM_scrubbing/train_data/train_ts2012_chembl20_MWmax800_scrubDM_minpos10_cutoff5.hdf5";
sea_lib=$HOME_SAVE_BASE"/output/20190410_SMA_Investigation/SEA_GEN/binding/chembl20_binding_ecfp4_4096.sea";

# output files
out_base=$HOME_SAVE_BASE"/output/20190410_SMA_Investigation/SEA_GEN/binding/preds";
mkdir -p $out_base;
log_file=$out_base"/sea_hdf5_preds.log"; 
out_file=$out_base"/train_ts2012_chembl20_MWmax800_scrubDM_minpos10_cutoff5_SEA_predictions.csv";

# log info
# reference files
the_deets="lab-notebook-caceres/Projects/nnets/20190410_SMA_Investigation/SEA_GEN";
right_now=$(date +"%x %r %Z"); 
curr_env="features"

# make our logfile
echo "Creating log file at "$log_file;
echo "Expt last run on $HOSTNAME" > $log_file; 
echo "Updated on $right_now by $USER" >> $log_file;
echo "For details and code for this experiment, please see: \n$the_deets" >> $log_file;
echo "Anaconda env used: $curr_env" >> $log_file;

# activate environment
echo "Activating environment...";
source activate $curr_env;

# source set nnet path
NNET_BASE=$HOME"/labgits/neural-nets";
source ${NNET_BASE}/set_nnet_path;

# run expt
echo "Making SEA predictions...";

run_preds="time python $make_predictions $hdf5_file $sea_lib $out_file --multi_instance > $log_file 2>&1;"

echo "${run_preds}"
eval ${run_preds} 