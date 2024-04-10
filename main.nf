params.genie_config = "${baseDir}/scripts/genie_config.json"
params.input_csv_path = "input_data.csv"

// def py_interpreter = "/home/jugo/anaconda3/envs/cgas-sting/bin/python"
def py_interpreter = "python3"

process filterRows {

    container 'bskracic/genie-runtime:latest'

    input:
    path input_csv

    output:
    file "filtered_data.csv"

    script:
    """
    ${py_interpreter} ${baseDir}/scripts/filter_rows.py --genie_config=${params.genie_config} --output_csv=filtered_data.csv
    """
}

process generateDataset {

    input:
    file filtered_data_csv
    path input_csv

    output:
    file "dataset.obj"

    script:
    """
    ${py_interpreter} ${baseDir}/scripts/generate_dataset.py --genie_config=${params.genie_config}  --input_csv=$filtered_data_csv --dataset_obj=dataset.obj
    """
}

process gaSearch {
    input:
    file "dataset.obj"
    path input_csv

    script:
    """
    ${py_interpreter} ${baseDir}/scripts/ga_search.py --dataset_obj=dataset.obj
    """
}

workflow {
    input_csv = channel.fromPath(params.input_csv_path)
    file = filterRows(input_csv)
    dataset = generateDataset(file, input_csv)
    gaSearch(dataset, input_csv)
}