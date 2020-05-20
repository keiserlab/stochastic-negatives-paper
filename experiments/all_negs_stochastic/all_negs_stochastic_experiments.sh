#!/bin/bash

# load gpu availablity functions
. /srv/home/nmew/myprojects/neural-nets/scripts/nvidia-available-gpus.sh

sneg_pnrs=(1.0 1.2222 0.8182 1.5 19.0 9.0 4.0 2.3333 0.6666 0.4286 0.25 0.1111 0.0753)
train_script="/srv/home/nmew/myprojects/clean-neural-nets/experiments/all_negs_stochastic/all_negs_stochastic.py"
sneg_arg_file="/fast/disk0/nmew/output/no_negatives/sneg_args.txt"

base_out_dir="/fast/disk0/nmew/output/no_negatives"
fold=$1
out_dir="${base_out_dir}/fold_${fold}"
test_inds="${base_out_dir}/kfold_indices/pickleKF_${fold}_indices"

for pnr in ${sneg_pnrs[@]}; do
  # no SEA
  output_dir="${out_dir}/sneg/pnr_${pnr}"
  output_weights="${output_dir}/model_at_epoch_199.npz"
  if [[ ! -e "$output_dir" ]] || [[ ! -e "$output_weights" ]]; then
    resume=""
    if [[ -e "$output_dir" ]]; then
      resume="--resume"
    fi
    mkdir -p ${output_dir}
    gpu="$( wait_for_available_gpu 1 )"
    set -x
    CUDA_VISIBLE_DEVICES=${gpu} python ${train_script} @${sneg_arg_file} --test_index_file ${test_inds} --positive_negative_ratio ${pnr} --output_directory ${output_dir} ${resume} &> ${output_dir}/log${resume}.txt &
    pid=$!
    set +x
    echo "${pnr} pid: ${pid}"
    sleep 15s
  fi
done

/srv/home/nmew/myprojects/neural-nets/scripts/gen_all_validation_analysis_for_current_dir.sh ${out_dir}

