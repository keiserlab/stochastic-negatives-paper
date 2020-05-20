#!/bin/sh
# Generate sneg and no-sneg experiments
# Contact: Elena Caceres (ecaceres@keiserlab.org) or Nick Mew (mew@keiserlab.org)
# Version: 1

# load gpu availablity functions
. /srv/home/ecaceres/nvidia-available-gpus.sh

# Run commands
NNET_BASE=$HOME"/labgits/neural-nets";
NNET_SCRIPT_BASE=$NNET_BASE"/scripts";
val_script=$NNET_SCRIPT_BASE"/gen_all_validation_analysis_for_current_dir.sh";

# where nnet architectures are stored
nn_basedir_nick="/srv/home/nmew/myprojects/clean-neural-nets";

# input variables
# where I saved the kfolds
expt_base=$CURRENT_SAVE_PLACE"/20190410_SMA_Investigation";
kfolds_dir="${expt_base}/kfold_indices";
# where the args files are stored for nnet training
args_path=$HOME"/labgits/lab-notebook-caceres/Projects/nnets/20190410_SMA_Investigation";

# output variables
save_dir="${CURRENT_SAVE_PLACE}/20190410_SMA_Investigation/trained_nets"; 
curr_env="mygpu1"

run_experiment () {

  fold="$1"   # 0,1,2,etc.
  train_script="$2"  # path to training script file
  sneg_nosneg="$3"   # either "sneg" or "no_sneg" should match argfile name eg. sneg_args.txt
  sneg_dir="$4"

  index_file="${kfolds_dir}/pickleKF_${fold}_indices"
  train_script_filename="$(basename "${train_script}")"
  train_script_name="${train_script_filename%.*}"
  outdir="${train_script_name}/${sneg_dir}/fold_${fold}"
  abs_outdir="${save_dir}/${outdir}"
  logfile="${abs_outdir}/train_log.txt"

  mkdir -p ${abs_outdir}


  # activate environment
  echo "Activating environment...";
  . activate ${curr_env};
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
  echo $CONDA_DEFAULT_ENV

  gpu="$( wait_for_available_gpu 1 )"
  reserve_gpu="CUDA_VISIBLE_DEVICES=${gpu}"
  try_run="THEANO_FLAGS=device=gpu${gpu}"

  train="python ${train_script} @${args_path}/${sneg_nosneg}_args.txt -i ${index_file} -o ${abs_outdir} &> ${logfile}"  
  validate="${val_script} ${abs_outdir} &> ${abs_outdir}/validation.log"  
  train_val="(THEANO_FLAGS=device=gpu ${reserve_gpu} ${train} ; ${validate} ; sleep 15m ;)"

  echo -e "\n\nrunning the following:"
  echo "${train_val}"
  eval ${train_val} &
  sleep 15s

}
 
for fold in {0..4}; do
  # std networks
  run_experiment ${fold} ${NNET_BASE}/experiments/lr_nesterov_1024_2048_3072.py no_sneg STD
  run_experiment ${fold} ${NNET_BASE}/experiments/lr_nesterov_1024_2048_3072.py sneg STD_SMA
  
  # Neg RM experiments
  run_experiment ${fold} ${NNET_BASE}/experiments/all_negs_stochastic/all_negs_stochastic.py no_sneg NEG_RM  
  run_experiment ${fold} ${NNET_BASE}/experiments/all_negs_stochastic/all_negs_stochastic.py sneg  NEG_RM_SMA

  # Neg UW expt
  run_experiment ${fold} ${NNET_BASE}/experiments/weighted_loss/weight_known_negative_loss.py no_sneg NEG_UW

done