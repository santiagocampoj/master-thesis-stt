#!/bin/bash

if [[ $CONDA_DEFAULT_ENV != "stt" ]]; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate stt
fi

use_nohup=false
MAX_JOBS=10  # Maximum number of simultaneous jobs

while getopts ":p:n" opt; do
  case $opt in
    p)
      base_directory="$OPTARG"
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

if [ -z "$base_directory" ]; then
    echo "Please provide the base directory using the -p option."
    exit 1
fi

for date_sub_directory in "$base_directory"/*; do
    if [ -d "$date_sub_directory" ]; then
        for sub_directory in "$date_sub_directory"/*; do
            if [ -d "$sub_directory" ]; then
                if $use_nohup; then
                    while (( $(jobs | wc -l) >= $MAX_JOBS )); do
                        sleep 1
                    done
                    nohup python3 -m Parlamento_EJ.main -d "$sub_directory" > /home/aholab/santi/Documents/audio_process/Language/Spanish/Parlamento_EJ/parlamento_ej.log 2>&1 &
                else
                    python3 -m Parlamento_EJ.main -d "$sub_directory"
                fi
            fi
        done
    fi
done
