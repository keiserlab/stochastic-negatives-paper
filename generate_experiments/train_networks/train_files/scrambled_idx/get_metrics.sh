#!/bin/sh
# Contact: Elena Caceres (ecaceres@keiserlab.org) or Nick Mew (mew@keiserlab.org)
# Version: 1

# load gpu availablity functions
. $HOME/nvidia-available-gpus.sh

# Run commands
NNET_BASE=$HOME"/labgits/neural-nets";
NNET_SCRIPT_BASE=$NNET_BASE"/scripts";
val_script=$NNET_SCRIPT_BASE"/gen_all_validation_analysis_for_current_dir.sh";

# where nnet architectures are stored
nn_basedir_nick="/srv/home/nmew/myprojects/clean-neural-nets";

# where I saved the kfolds
expt_base=$DATA_SAVE_BASE"/20180525_DM_scrubbing/train_data";
kfolds_dir="${expt_base}/kfold_indices";

# where the args files are stored for nnet training
args_path=$HOME"/labgits/lab-notebook-caceres/Projects/nnets/20180815_Paper_Retrains/scrambled_idx";

# output variables
save_dir="${HOME_SAVE_BASE}/output/20180815_Paper_Retrains/trained_nets"; 
curr_env="mygpu1"

PRED_THRESH="5.0"
REGRESSION="True"

# activate environment
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

run_experiment () {

  fold="$1"   # 0,1,2,etc.
  train_script="$2"  # path to training script file
  pnr="$3"   # either "sneg" or "no_sneg" should match argfile name eg. sneg_args.txt
  sneg_dir="$4"
  

  index_file="${kfolds_dir}/pickleKF_${fold}_indices"
  train_script_filename="$(basename "${train_script}")"
  train_script_name="${train_script_filename%.*}"
  outdir="${train_script_name}/${sneg_dir}/fold_${fold}"
  abs_outdir="${save_dir}/${outdir}"
  
  output_dir="${abs_outdir}/pnr_${pnr}"
  validate="${val_script} ${output_dir} ${PRED_THRESH} ${REGRESSION} &> ${output_dir}/validation.log & pid=$!"

  # define train script
  val_run="(THEANO_FLAGS=device=gpu ${reserve_gpu} ${validate}; sleep 5m;)"

  echo -e "\n\nrunning the following:"
  echo "${pnr} pid: ${pid}"
  echo "${val_run}"
  eval ${val_run} &
  sleep 15s
  
}

sneg_pnrs=(1.0)

for fold in {0..4}; do
  for pnr in ${sneg_pnrs[@]}; do
    run_experiment ${fold} ${NNET_BASE}/experiments/lr_nesterov_1024_2048_3072.py ${pnr} scrambled_idx
  done
done
