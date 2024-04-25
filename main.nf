params.genie_config_path = "${baseDir}/scripts/genie_config.json"
params.target = "DSS"
params.wandb_api_key = "7378d591fc5cde507d409f0b843f8c0b95a6969b"
params.cancers = 'BRCA,LUAD'

//def py_interpreter = "/home/jugo/anaconda3/envs/cgas-sting/bin/python"
def py_interpreter = "python3"

process filterRows {

    container 'bskracic/genie-runtime:latest'

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

process gaSearch {
    input:
    file "dataset.obj"
    path genie_config_path

    output:
    file "adj_matrix.obj"
    file "wandb_run_id.txt"

    script:
    """
    ${py_interpreter} ${baseDir}/scripts/ga_search.py --genie_config=${params.genie_config_path} --cancers=${params.cancers} --dataset_obj=dataset.obj --adj_matrix_obj=adj_matrix.obj --wandb_api_key=${params.wandb_api_key} --target=${params.target}
    """
}

process integrated_gradients {
    input:
    file "wandb_run_id.txt"
    file "adj_matrix.obj"
    file "dataset.obj"
    path genie_config_path

    script:
    """
    ${py_interpreter} ${baseDir}/scripts/integrated_gradients.py --genie_config=${params.genie_config_path} --target=${params.target} --dataset_obj=dataset.obj --adj_matrix_obj=adj_matrix.obj --wandb_api_key=${params.wandb_api_key}
    """
}

workflow {
    genie_config_path = channel.fromPath(params.genie_config_path)
    file = filterRows(genie_config_path)
    dataset = generateDataset(file, genie_config_path)
    gaSearch(dataset, genie_config_path)
    adj_matrix = gaSearch.out[0]
    wandb_run_id_txt = gaSearch.out[1]
    integrated_gradients(wandb_run_id_txt, adj_matrix, dataset, genie_config_path)
}