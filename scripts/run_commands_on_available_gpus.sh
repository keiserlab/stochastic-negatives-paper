#!/bin/bash

# load gpu availablity functions
. /srv/home/nmew/myprojects/neural-nets/scripts/nvidia-available-gpus.sh

python_scripts_file="$1"
procs_per_gpu="1"
use_cpu="use_gpu"
if [[ $# -gt 1 ]]; then
  procs_per_gpu="$2"
fi
if [[ $# -gt 2 ]]; then
  if [[ "$3" == "use_cpu" ]]; then 
    use_cpu="use_cpu"
  fi
fi

if [[ $# -lt 1 ]]; then
echo "${BASH_SOURCE[0]} python_scripts_with_args [num_processes_per_gpu] ['use_cpu']"
exit 1
fi


# read list of command lines from file
readarray python_scripts <  ${python_scripts_file}
num_py_scripts=$(wc -l  ${python_scripts_file} | cut -d' ' -f1)
ind=0
while [[ $ind -lt $num_py_scripts ]]; do
  train_script="${python_scripts[ind]}"
  ind=$((ind + 1))
  if [[ "$use_cpu" != "use_cpu" ]]; 
    then
      next_available_gpu="$( wait_for_available_gpu $procs_per_gpu )"
      device="CUDA_VISIBLE_DEVICES=${next_available_gpu} "
    else
      device="THEANO_FLAGS='device=cpu' "
  fi
  train_script_cl="${device} ${train_script} &"
  echo "${train_script_cl}"
  eval ${train_script_cl} 
  pid=$!
  echo "pid: ${pid}"
  sleep 30s
done

