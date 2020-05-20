#!/bin/sh
# Contact: Elena Caceres (ecaceres@keiserlab.org) or Nick Mew (mew@keiserlab.org)
# Version: 1

# load gpu availablity functions
. $HOME/nvidia-available-gpus.sh

# Experiment json file
expt_name="CLASSIFIER_SEA_SMA_LR03";
organizing_folder="20190410_SMA_Investigation";
converged_epochs_json="${DATA_SAVE_BASE}/${organizing_folder}/predictions/${expt_name}/experiments.json";

# Run commands
NNET_BASE=$HOME"/labgits/neural-nets";
NNET_SCRIPT_BASE=$NNET_BASE"/common";
get_preds_script=$NNET_SCRIPT_BASE"/get_preds.py";

# input args **CHANGE TRAIN FOR SCRAMBLED**
data_dir="${DATA_SAVE_BASE}/20180525_DM_scrubbing/train_data/"
hdf5_train="${data_dir}train_ts2012_chembl20_MWmax800_scrubDM_minpos10_cutoff5.hdf5";
hdf5_ts="${data_dir}val_ts2012_chembl20_MWmax800_scrubDM_minpos10_cutoff5.hdf5";
target_map_file="${data_dir}ts2012_chembl20_MWmax800_scrubDM_minpos10_cutoff5_target_index.pkl";
out_dir="${DATA_SAVE_BASE}/${organizing_folder}/predictions/${expt_name}/"; 
network_script_fmter="${HOME}/labgits/neural-nets/experiments/classification/{}.py";

# make folders as necessary
mkdir -p "${out_dir}/targets";

# activate environment
curr_env="mygpu1"
echo "Activating environment...";
. activate ${curr_env};
# set cuda env 
. $HOME/.gpu_profile

# set cuda profile
if [ "$HOSTNAME" == "mk-gpu-2.c.keiserlab.org" ]; then 
  . /fast/ssd0/.cuda_profile
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
else
  . /fast/disk0/shared/.cuda_profile
fi

# source set nnet path
. ${NNET_BASE}/set_nnet_path
echo $CONDA_DEFAULT_ENV

pred_call="python ${get_preds_script} ${hdf5_train} ${hdf5_ts} ${target_map_file} ${out_dir} ${converged_epochs_json} --network_script_fmter ${network_script_fmter} &> ${out_dir}/get_preds.log & pid=$!"3

gpu="$( wait_for_available_gpu 1 )"
reserve_gpu="CUDA_VISIBLE_DEVICES=${gpu}"

# define train script
preds_run="(THEANO_FLAGS=device=gpu ${reserve_gpu} ${pred_call}; sleep 5m;)"

echo -e "\n\nrunning the following:"
echo "pid: ${pid}"
echo "${preds_run}"
eval ${preds_run} &
sleep 15s