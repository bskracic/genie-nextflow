#!/bin/bash

# cancers=('BRCA' 'LUNG' 'KIRC' 'HNSC' 'LUAD' 'THCA' 'LUSC' 'SKCM' 'STAD' 'BLCA')
cancers=('HNSC' 'SKCM')
# BLCA is missing
targets=('DSS' 'OS' 'GENDER' 'STAGE')

for cancer in "${cancers[@]}"; do
    for target in "${targets[@]}"; do
      cmd="
        ./nextflow -log nextflow-logs/log run main.nf \\
          --genie_config_path=\"/home/jugo/research/genie-ig-rerun/scripts/genie_config.json\" \\
          --wandb_api_key=7378d591fc5cde507d409f0b843f8c0b95a6969b \\
          --cancers=\"${cancer}\" --target=\"${target}\"
        "
      $cmd
    done
done