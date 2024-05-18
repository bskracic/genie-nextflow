#!/bin/bash

#cancers=('HNSC' 'KIRC' 'LGG')
cancers=('SKCM')
targets=('OS' 'GENDER' 'STAGE')
#targets=('DSS')

for cancer in "${cancers[@]}"; do
    for target in "${targets[@]}"; do
      cmd="
        ./nextflow -log nextflow-logs/log run main.nf \\
          --genie_config_path=\"\$(pwd)/scripts/genie_config.json\" \\
          --wandb_api_key=7378d591fc5cde507d409f0b843f8c0b95a6969b \\
          --cancers=\"${cancer}\" --target=\"${target}\"
        "
      gnome-terminal -- bash -c "$cmd" &
    done
    wait
done
