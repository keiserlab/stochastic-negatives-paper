#!/bin/bash

echo "One Script to rule them all, One Script to find them,"
sleep 1
echo "One Script to bring them all and in the darkness bind them"
sleep 1


get_next_job(){
  # Read first line and remove from commands file
  command=$(head -1 $input_scripts_file)
  echo "$(tail -n +2 $input_scripts_file)" > $input_scripts_file
  echo $command
}

start_job(){
  # Start the job (from subshell cwd) and output to log
  command=$1
  logfile=job_`date +%Y.%m.%d-%H.%M.%S`.log
  (cd $command_path;
  eval "$command" > "$log_path/$logfile" 2>&1 &)

  # Get the pid
  job_pid=$!

  # Add job and information to started jobs file
  echo $'\n'`date` >> $output_scripts_file
  echo "Started job with pid $job_pid, logging to $logfile" >> $output_scripts_file
  echo $command >> $output_scripts_file

  # Print the same information
  echo $'\n'`date`
  echo "Started job with pid $job_pid, logging to $logfile"
  echo $command
}

get_available_gpu(){
  echo $(/srv/home/nmew/myprojects/neural-nets/scripts/nvidia-available-gpus.sh | head -1)
}

# Set paths
input_scripts_file=$1
command_path=${2:-$(pwd)}
output_scripts_file=${3:-${input_scripts_file%.*}.started.${input_scripts_file##*.}}
log_path=${4:-$(pwd)}

echo $'\n'"====================================="
echo "       QUE-IN: $input_scripts_file"
echo "      QUE-OUT: $output_scripts_file"
echo " COMMAND-PATH: $command_path"
echo "     LOG-PATH: $log_path"

echo $'\n'`date` >> $output_scripts_file
echo "Started job starter with paths:" >> $output_scripts_file
echo "       QUE-IN: $input_scripts_file" >> $output_scripts_file
echo "      QUE-OUT: $output_scripts_file" >> $output_scripts_file
echo "  SCRIPT-PATH: $(pwd)" >> $output_scripts_file
echo " COMMAND-PATH: $command_path" >> $output_scripts_file
echo "     LOG-PATH: $log_path" >> $output_scripts_file

# Get first job
command=$(get_next_job)

i=0
spin='-\|/'

# While there are still jobs left to do
while [[ !(-z $command) ]]; do

  # Wait for a gpu to become available
  next_available_gpu=$(get_available_gpu)
  while [[ -z $next_available_gpu ]]; do
    i=$(( (i+1) %4 ))
    printf "\rWaiting for available gpu ${spin:$i:1}"
    sleep 10
    next_available_gpu=$(get_available_gpu)
  done

  # Start job and get next job
  start_job "CUDA_VISIBLE_DEVICES=${next_available_gpu} $command"
  command=$(get_next_job)

  # Sleep to let the job claim GPU resources (and ensure different logfile)
  sleep 60
done