/home/bskracic/miniconda3/envs/genie/bin/python3 scripts/filter_rows.py \
    --genie_config=scripts/genie_config.json --cancers=LUSC --target=DSS --output_csv dumps/BRCA_dataset_dump.csv

/home/bskracic/miniconda3/envs/genie/bin/python3 scripts/generate_dataset.py \
    --genie_config=scripts/genie_config.json --input_csv=dumps/BRCA_dataset_dump.csv --dataset_obj=dumps/BRCA_dataset.obj

/home/bskracic/miniconda3/envs/genie/bin/python3 scripts/ig_review.py \
    --genie_config=scripts/genie_config.json \
    --input_csv=dumps/BRCA_dataset_dump.csv \
    --dataset_obj=dumps/BRCA_dataset.obj \
    --cancers=BRCA \
    --target=DSS \
    --wandb_api_key=7378d591fc5cde507d409f0b843f8c0b95a6969b \
    --adj_matrix=adj-matrix/BRCA/DSS-adj-matrix.obj


