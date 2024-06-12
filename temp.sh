/home/jugo/anaconda3/envs/cgas-sting/bin/python3 scripts/filter_rows.py \
    --genie_config=scripts/genie_config.json --cancers=LUSC --target=DSS --output_csv dumps/LUSC_dataset_dump.csv

/home/jugo/anaconda3/envs/cgas-sting/bin/python3 scripts/generate_dataset.py \
    --genie_config=scripts/genie_config.json --input_csv=dumps/LUSC_dataset_dump.csv --dataset_obj=dumps/LUSC_dataset.obj