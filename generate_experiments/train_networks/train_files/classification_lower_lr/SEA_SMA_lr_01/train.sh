#!/bin/sh
# Generate sneg and no-sneg experiments
# Contact: Elena Caceres (ecaceres@keiserlab.org) or Nick Mew (mew@keiserlab.org)
# Version: 1

# load gpu availablity functions
. /srv/home/ecaceres/nvidia-available-gpus.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID

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
args_path=$HOME"/labgits/lab-notebook-caceres/Projects/nnets/20180815_Paper_Retrains/classification_lower_lr/SEA_SMA_lr_01";

# output variables
save_dir="${HOME_SAVE_BASE}/output/20180815_Paper_Retrains/trained_nets"; 
curr_env="mygpu1"

# activate environment
echo "Activating environment...";
. activate ${curr_env};
# set cuda env 
. $HOME/.gpu_profile
# . /fast/disk0/shared/.cuda_profile

# set cuda env 
. /fast/ssd0/.cuda_profile
# source set nnet path
. ${NNET_BASE}/set_nnet_path

run_experiment () {
  fold="$1"   # 0,1,2,etc.
  train_script="$2"  # path to training script file
  sneg_nosneg="$3"   # either "sneg" or "no_sneg" should match argfile name eg. sneg_args.txt
  sneg_dir="$4"
  pnr="1.0"
  
  index_file="${kfolds_dir}/pickleKF_${fold}_indices"
  train_script_filename="$(basename "${train_script}")"
  train_script_name="${train_script_filename%.*}"
  outdir="${train_script_name}/${sneg_dir}/fold_${fold}"
  abs_outdir="${save_dir}/${outdir}"

  echo $CONDA_DEFAULT_ENV
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
    gpu="$( wait_for_available_gpu 2 )"
    reserve_gpu="CUDA_VISIBLE_DEVICES=${gpu}"
  
    # define train script
    train="python ${train_script} @${args_path}/${sneg_nosneg}_args.txt -i ${index_file} -o ${output_dir} ${resume} &> ${output_dir}/log${resume}.txt & 
    pid=$!"
    train_run="(THEANO_FLAGS=device=gpu ${reserve_gpu} ${train} ; sleep 15m ; )"
  
    echo -e "\n\nrunning the following:"
    echo "${pnr} pid: ${pid}"
    echo "${train_run}"
    eval ${train_run} &
    sleep 15s;
  fi
  }

for fold in {0..4}; do
  run_experiment ${fold} ${NNET_BASE}/experiments/classification/lr_nesterov_binary_classifier_1024_2048_3072.py classifier_sea_sneg CLASSIFIER_SEA_SMA_LR01
done
