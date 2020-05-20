#!/bin/bash

nvidia-available-gpus () {

  if [ $# -gt 0 ]; then
    max_jobs_per_gpu="$1"
  else
    max_jobs_per_gpu=1
  fi

  # max_jobs_per_gpu=$((max_jobs_per_gpu + 1))
  gpus_in_use="$(nvidia-smi --format=csv,noheader --query-compute-apps=gpu_bus_id)"
  gpu_indexes_in_use="$(for bus in ${gpus_in_use[@]}; do echo "$(nvidia-smi --format=csv,noheader --query-gpu=gpu_bus_id,index | grep ${bus} | cut -d' ' -f2)"; done)"

  # echo ${gpu_indexes_in_use[@]} $(nvidia-smi --format=csv,noheader --query-gpu=index) | tr ' ' '\n' | sort | uniq -u
  echo ${gpu_indexes_in_use[@]} $(nvidia-smi --format=csv,noheader --query-gpu=index) | tr ' ' '\n' | sort | uniq -c | grep " [0-${max_jobs_per_gpu}] " | awk '{print $2}' 
}

next_available_gpu () {
  if [ $# -gt 0 ]; then
    max_jobs_per_gpu="$1"
  else
    max_jobs_per_gpu=1
  fi

  echo "$( nvidia-available-gpus ${max_jobs_per_gpu} | head -1)"
}


wait_for_available_gpu () {
  if [ $# -gt 0 ]; then
    max_jobs_per_gpu="$1"
  else
    max_jobs_per_gpu=1
  fi
  available_gpu=$( next_available_gpu ${max_jobs_per_gpu} )
  if [[ -z $available_gpu ]]; then echo "no gpu available" >&2 ; fi
  while [[ -z $available_gpu ]]; do
    sleep 10s
    echo -n "." >&2
    available_gpu=$( next_available_gpu ${max_jobs_per_gpu} )
  done
  echo ${available_gpu}
}

if [ $# -gt 0 ]; then
  max_jobs_per_gpu="$1"
else
  max_jobs_per_gpu=1
fi

nvidia-available-gpus ${max_jobs_per_gpu}

