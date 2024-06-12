params.genie_config_path = "${baseDir}/scripts/genie_config.json"
params.target = "DSS"
params.wandb_api_key = "7378d591fc5cde507d409f0b843f8c0b95a6969b"
params.cancers = 'BRCA'

def py_interpreter = "/home/jugo/anaconda3/envs/cgas-sting/bin/python3"

process filterRows {

    input:
    path genie_config_path

    output:
    file "filtered_data.csv"

    script:
    """
    ${py_interpreter} ${baseDir}/scripts/filter_rows.py --genie_config=${params.genie_config_path} --cancers=${params.cancers} --output_csv=filtered_data.csv --target=${params.target}
    """
}

process generateDataset {

    input:
    file filtered_data_csv
    path genie_config_path

    output:
    file "dataset.obj"

    script:
    """
    ${py_interpreter} ${baseDir}/scripts/generate_dataset.py --genie_config=${params.genie_config_path}  --input_csv=$filtered_data_csv --dataset_obj=dataset.obj
    """
}

process integrated_gradients {
    input:
    file "dataset.obj"
    file filtered_data_csv
    path genie_config_path

    script:
    """
    ${py_interpreter} ${baseDir}/scripts/ig_review.py --genie_config=${params.genie_config_path} --target=${params.target} --dataset_obj=dataset.obj --wandb_api_key=${params.wandb_api_key} --adj_matrix=${baseDir}/adj-matrix/${params.cancers}/${params.target}-adj-matrix.obj --target=${params.target} --cancer=${params.cancers} --input_csv=$filtered_data_csv
    """
}

workflow {
    genie_config_path = channel.fromPath(params.genie_config_path)
    file = filterRows(genie_config_path)
    dataset = generateDataset(file, genie_config_path)
    integrated_gradients(dataset, file, genie_config_path)
}