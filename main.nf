params.cancer_type = "desired_cancer_type"
params.input_csv_path = "input_data.csv"
params.input_genes = "genes.txt"
params.target_variable = "DSS"
params.clean_data_path = "clean_data.txt"

// def py_interpreter = "/home/jugo/anaconda3/envs/cgas-sting/bin/python"
def py_interpreter = "python3"

process filterRows {

    container 'bskracic/genie-runtime:latest'

    input:
    path input_csv
    path clean_data

    output:
    file "filtered_data.csv"

    script:
    """
    ${py_interpreter} ${baseDir}/scripts/filter_rows.py --input_csv $input_csv --clean_data $clean_data --output_csv filtered_data.csv --cancer_type ${params.cancer_type} --target ${params.target_variable}
    """
}

process trainModel {

    input:
    file filtered_data_csv

    script:
    """
    cat $filtered_data_csv
    """
}

workflow {
    input_csv = channel.fromPath(params.input_csv_path)
    clean_data = channel.fromPath(params.clean_data_path)
    file = filterRows(input_csv, clean_data)
    trainModel(file)
}