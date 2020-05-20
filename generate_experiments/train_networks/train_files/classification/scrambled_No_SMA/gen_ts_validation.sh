#!/bin/sh
# Generate time split analysis for expt
# Contact: Elena Caceres (ecaceres@keiserlab.org) or Nick Mew (mew@keiserlab.org)
# Version: 1

# load gpu availablity functions
. /srv/home/ecaceres/nvidia-available-gpus.sh

# Run commands
NNET_BASE=$HOME"/labgits/neural-nets";
NNET_SCRIPT_BASE=$NNET_BASE"/scripts";
val_script=$NNET_SCRIPT_BASE"/gen_newts_validation_for_dir.sh";

# where nnet architectures are stored
nn_basedir_nick="/srv/home/nmew/myprojects/clean-neural-nets";

# where I saved the kfolds
expt_base=$DATA_SAVE_BASE"/20180525_DM_scrubbing/train_data";
kfolds_dir="${expt_base}/kfold_indices";

# output variables
save_dir="${HOME_SAVE_BASE}/output/20180815_Paper_Retrains/trained_nets"; 
curr_env="mygpu1"

# input files

ts_file="${expt_base}/val_ts2012_chembl20_MWmax800_scrubDM_minpos10_cutoff5.hdf5";
ts_targs="${expt_base}/ts2012_chembl20_MWmax800_scrubDM_minpos10_cutoff5_target_index.pkl";

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
  pnr="$3"   # either "sneg" or "no_sneg" should match argfile name eg. sneg_args.txt
  sneg_dir="$4"
  

  index_file="${kfolds_dir}/pickleKF_${fold}_indices"
  train_script_filename="$(basename "${train_script}")"
  train_script_name="${train_script_filename%.*}"
  outdir="${train_script_name}/${sneg_dir}/fold_${fold}"
  abs_outdir="${save_dir}/${outdir}"
  output_dir="${abs_outdir}/pnr_${pnr}"
  
  validate="${val_script} ${output_dir} ${ts_file} ${ts_targs} &> ${output_dir}/ts_validation.log & pid=$!"  
  
  # reserve gpu
  gpu="$( wait_for_available_gpu 1 )"
  reserve_gpu="CUDA_VISIBLE_DEVICES=${gpu}"
  
  # define train script
  val_run="(THEANO_FLAGS=device=gpu ${reserve_gpu} ${validate}; sleep 5m;)"

  echo -e "\n\nrunning the following:"
  echo "${pnr} pid: ${pid}"
  echo "${val_run}"
  eval ${val_run} &
  sleep 15s
}
    
for fold in {0..4}; do
    run_experiment ${fold} ${NNET_BASE}}/experiments/lr_nesterov_1024_2048_3072.py 1.0 CLASSIFIER_scrambled_idx_no_SMA
done

. deactivate