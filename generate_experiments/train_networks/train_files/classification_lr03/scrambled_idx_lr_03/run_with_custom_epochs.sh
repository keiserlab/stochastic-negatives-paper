#!/bin/sh
# Generate sneg and no-sneg experiments
# Contact: Elena Caceres (ecaceres@keiserlab.org) or Nick Mew (mew@keiserlab.org)
# Version: 1

# load gpu availablity functions
. /srv/home/ecaceres/nvidia-available-gpus.sh

NNET_BASE=$HOME"/labgits/neural-nets";
# where the args files are stored for nnet traißning
args_path=$HOME"/labgits/lab-notebook-caceres/Projects/nnets/20180815_Paper_Retrains/classification_lr03/scrambled_idx_lr_03";

curr_env="mygpu1"

# set cuda env 
. $HOME/.gpu_profile

# set cuda profile
if [ "$HOSTNAME" == "mk-gpu-2.c.keiserlab.org" ]; then 
  . /fast/ssd0/.cuda_profile
else
  . /fast/disk0/shared/.cuda_profile
fi

# source set nnet path
. ${NNET_BASE}/set_nnet_path

# activate environment
echo "Activating environment...";
. activate ${curr_env};

echo $CONDA_DEFAULT_ENV

gpu="$( wait_for_available_gpu 1 )"
reserve_gpu="CUDA_VISIBLE_DEVICES=${gpu}"

try_run="THEANO_FLAGS=device=gpu${gpu}"

script_run="python ${args_path}/run_with_custom_epochs.py"  

run_expt="(THEANO_FLAGS=device=gpu ${reserve_gpu} ${script_run}; sleep 15m;)"

echo -e "\n\nrunning the following:"
echo "${run_expt}"
eval ${run_expt} &
sleep 15s
. deactivate
