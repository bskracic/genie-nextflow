import argparse
import re
import sys

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from genie_utils import load_config


def transform_id(value: str):
    parts = value.split("-")
    if len(parts) < 4:
        return '-'.join(parts)
    else:
        return '-'.join(parts[:-1])


def transform(row):
    value = row['ajcc_pathologic_tumor_stage']
    if value in ['[Not Applicable]', '[Not Available]', '[Unknown]', '[Discrepancy]', 'I/II NOS', 'Stage X']:
        value = row['clinical_stage']

    stage_1_pattern = re.compile(r'^(Stage\s)?(I(?:[aA]|[^IV])*)$')
    stage_2_pattern = re.compile(r'^(Stage\s)?(II(?:[aA]|[^IV])*)$')
    stage_3_pattern = re.compile(r'^(Stage\s)?(III(?:[aA]|[^IV])*)$')
    stage_4_pattern = re.compile(r'^(Stage\s)?(IV(?:[aA]|.)*)$')

    if 'IS' in value:
        return 'Stage 0'
    elif stage_1_pattern.match(value):
        return 'Stage 1'
    elif stage_2_pattern.match(value):
        return 'Stage 2'
    elif stage_3_pattern.match(value):
        return 'Stage 3'
    elif stage_4_pattern.match(value):
        return 'Stage 4'

    return value


def prepare_stage(df: pd.DataFrame):
    df['ajcc_pathologic_tumor_stage'] = df.apply(transform, axis=1)
    df = df[
        df.apply(lambda row: row['ajcc_pathologic_tumor_stage'] not in ['[Not Applicable]', '[Not Available]'], axis=1)]
    df = df.reset_index(drop=True)
    encoder = OneHotEncoder(sparse=False)
    encoded_array = encoder.fit_transform(df[['ajcc_pathologic_tumor_stage']])
    columns = [str.replace(val, 'ajcc_pathologic_tumor_stage_', '') for val in
               encoder.get_feature_names_out(['ajcc_pathologic_tumor_stage'])]
    out_encoded_df = pd.DataFrame(encoded_array, columns=columns)
    df = df.drop(['ajcc_pathologic_tumor_stage'], axis=1)
    df = pd.concat([df, out_encoded_df], axis=1)

    return df


def read_gene_file(file: str, genes: list) -> pd.DataFrame:
    temp_df: pd.DataFrame = pd.read_csv(file, sep='\t', header=None)
    temp_df = temp_df.T
    temp_df.columns = temp_df.iloc[0]
    temp_df = temp_df[1:]
    temp_df.rename(columns={'sample': 'patient_id'}, inplace=True)
    temp_df.set_index(pd.Index(range(len(temp_df))), inplace=True)
    temp_df = temp_df.dropna()
    temp_df = temp_df[['patient_id'] + genes]
    temp_df[genes] = temp_df[genes].astype(float)
    temp_df.loc[:, 'patient_id'] = temp_df['patient_id'].apply(transform_id)
    original_columns = temp_df.columns
    temp_df = temp_df.groupby('patient_id').agg(['mean'])
    temp_df = temp_df.reset_index()
    temp_df.columns = original_columns
    scaler = MinMaxScaler()
    temp_df[genes] = scaler.fit_transform(temp_df[genes])

    return temp_df


def load_dataset(config: dict, cancers: list[str]) -> pd.DataFrame:
    df = pd.DataFrame()

    for cancer in cancers:
        expression_path = config['cancers'][cancer]['expression']
        df = pd.concat([df, read_gene_file(expression_path, config['genes'])])

    survival_dataframe = pd.read_csv(config['survival_data_path'], sep='\t')
    survival_dataframe = survival_dataframe.drop(survival_dataframe.columns[0], axis=1)
    survival_dataframe = survival_dataframe.rename(columns={'bcr_patient_barcode': 'patient_id'})

    print(f'patients with gene expression data: {len(df)}')
    print(f'patients with survival data: {len(survival_dataframe)}')

    df = df.merge(survival_dataframe, on='patient_id', how='inner')

    print(f'patients after merging: {len(df)}')

    return df


outputs: dict = {
    'STAGE': ['early', 'late'],
    'DSS': ['DSS_1', 'DSS_0'],
    'OS': ['OS_1', 'OS_0'],
    'GENDER': ['gender_female', 'gender_male']
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter rows by cancer type/s')
    parser.add_argument('--genie_config', required=True, help='Genie json configuration file')
    parser.add_argument('--output_csv', required=True, help='Resulting filtered rows csv file')
    parser.add_argument('--cancers', required=True, help='Comma separated list of cancers')
    parser.add_argument('--target', required=True, help='Target variable')

    args = parser.parse_args()

    config = load_config(args.genie_config)
    target = args.target

    # Exit early if target is not in valid
    if target not in outputs.keys():
        sys.exit("WRONG TARGET VARIABLE")

    cancers = args.cancers.split(',')
    dataframe: pd.DataFrame = load_dataset(config, cancers)

    columns = config['genes']
    match target:
        case 'DSS':
            dataframe["DSS_1"] = dataframe.apply(lambda row: 1 if row['DSS'] == 1 else 0, axis=1)
            dataframe["DSS_0"] = dataframe.apply(lambda row: 1 if row['DSS'] == 0 else 0, axis=1)
            columns += ['DSS_1', 'DSS_0']
        case 'OS':
            dataframe["OS_1"] = dataframe.apply(lambda row: 1 if row['OS'] == 1 else 0, axis=1)
            dataframe["OS_0"] = dataframe.apply(lambda row: 1 if row['OS'] == 0 else 0, axis=1)
            columns += ['OS_1', 'OS_0']
        case 'GENDER':
            dataframe["gender_female"] = dataframe.apply(lambda row: 1 if row['gender'] == 'FEMALE' else 0, axis=1)
            dataframe["gender_male"] = dataframe.apply(lambda row: 1 if row['gender'] == 'MALE' else 0, axis=1)
            columns += ['gender_female', 'gender_male']
        case 'STAGE':
            dataframe = prepare_stage(dataframe)
            dataframe["early"] = dataframe.apply(lambda row: 1 if row['Stage 1'] == 1 or row['Stage 2'] == 1 else 0,
                                                 axis=1)
            dataframe["late"] = dataframe.apply(lambda row: 1 if row['Stage 4'] == 1 or row['Stage 3'] == 1 else 0,
                                                axis=1)
            columns += ["early", "late"]

    dataframe[columns].to_csv(args.output_csv, index=False)
