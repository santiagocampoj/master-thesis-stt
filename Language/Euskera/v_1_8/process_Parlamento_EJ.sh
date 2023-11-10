#!/bin/bash

if [[ $CONDA_DEFAULT_ENV != "stt" ]]; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate stt
fi

use_nohup=false

while getopts ":p:n" opt; do
  case $opt in
    p)
      parent_directory="$OPTARG"
      ;;
    n)
      use_nohup=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

if [ -z "$parent_directory" ]; then
    echo "Please provide the parent directory using the -p option."
    exit 1
fi

# echo "$parent_directory"

for sub_directory in "$parent_directory"/*; do
    if [ -d "$sub_directory" ]; then
        if $use_nohup; then
            nohup python3 -m Parlamento_EJ.main -d "$sub_directory" > /home/aholab/santi/Documents/audio_process/Language/Spanish/Parlamento_EJ/parlamento_ej.log 2>&1 &
        else
            python3 -m Parlamento_EJ.main -d "$sub_directory"
        fi
    fi
done
