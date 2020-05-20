#!/bin/sh
# Generate sneg and no-sneg experiments
# Contact: Elena Caceres (ecaceres@keiserlab.org) or Nick Mew (mew@keiserlab.org)
# Version: 1

# load gpu availablity functions
. /srv/home/ecaceres/nvidia-available-gpus.sh

# Run commands
NNET_BASE=$HOME"/labgits/neural-nets";
NNET_SCRIPT_BASE=$NNET_BASE"/scripts";

# where nnet architectures are stored
nn_basedir_nick="/srv/home/nmew/myprojects/clean-neural-nets";

# where I saved the kfolds
expt_base=$CURRENT_SAVE_PLACE"/20190410_SMA_Investigation";
kfolds_dir="${expt_base}/kfold_indices";

# where the args files are stored for nnet training
args_path=$HOME"/labgits/lab-notebook-caceres/Projects/nnets/20190410_SMA_Investigation";

# output variables
save_dir="${CURRENT_SAVE_PLACE}/20190410_SMA_Investigation/trained_nets"; 
curr_env="mygpu1"

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

run_experiment () {

  fold="$1"   # 0,1,2,etc.
  train_script="$2"  # path to training script file
  sneg_nosneg="$3"   # either "sneg" or "no_sneg" should match argfile name eg. sneg_args.txt
  sneg_dir="$4"
  
  #sneg_pnrs=(1.2222 0.8182 1.5 19.0 9.0 4.0 2.3333 0.6666 0.4286 0.25 0.1111 0.0753 1.0)
  sneg_pnrs=(1.0)
  index_file="${kfolds_dir}/pickleKF_${fold}_indices"
  train_script_filename="$(basename "${train_script}")"
  train_script_name="${train_script_filename%.*}"
  outdir="${train_script_name}/${sneg_dir}/fold_${fold}"
  abs_outdir="${save_dir}/${outdir}"

  for pnr in ${sneg_pnrs[@]}; do
    output_dir="${abs_outdir}/pnr_${pnr}"
    output_weights="${output_dir}/model_at_epoch_199.npz"
    # check to see if output directory exists and if .npz file exists, if not, run from scratch.
    if [[ ! -e "$output_dir" ]] || [[ ! -e "$output_weights" ]]; then
      resume=""
      if [[ -e "$output_dir" ]]; then
        resume="--resume"
      fi
      mkdir -p ${output_dir}
      # reserve GPU
      gpu="$( wait_for_available_gpu 1 )"
      reserve_gpu="CUDA_VISIBLE_DEVICES=${gpu}"
      
      # define train script
      train="python ${train_script} @${args_path}/${sneg_nosneg}_args.txt -i ${index_file} --positive_negative_ratio ${pnr} -o ${output_dir} ${resume} &> ${output_dir}/training_log${resume}.txt & 
      pid=$!"
      train_run="(THEANO_FLAGS=device=gpu ${reserve_gpu} ${train} ; sleep 15m ; )"
      
      echo -e "\n\nrunning the following:"
      echo "${pnr} pid: ${pid}"
      echo "${train_run}"
      eval ${train_run} &
      sleep 15s
      
    fi
  done
}
 
for fold in {0..4}; do
  #run_experiment ${fold} ${NNET_BASE}/experiments/lr_nesterov_1024_2048_3072.py sneg_scrambled STD_SMA_RATIOS_scrambled
  
  run_experiment ${fold} ${NNET_BASE}/experiments/all_negs_stochastic/all_negs_stochastic.py sneg_scrambled NEG_RM_RATIOS_scrambled
done
