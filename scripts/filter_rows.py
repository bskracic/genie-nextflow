import argparse
import sys

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from genie_utils import load_config


def transform_id(value: str):
    parts = value.split("-")
    if len(parts) < 4:
        return '-'.join(parts)
    else:
        return '-'.join(parts[:-1])


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


def load_dataset(config: dict) -> pd.DataFrame:
    df = pd.DataFrame()
    for cancer in config['cancers']:
        expression_path = cancer['files']['expression']
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
    'STAGE': ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'],
    'STAGE_EL': ['early', 'late'],
    'DSS': ['DSS_1', 'DSS_0'],
    'OS': ['OS_1', 'OS_0'],
    'GENDER': ['gender_female', 'gender_male']
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter rows by cancer type/s')
    parser.add_argument('--genie_config', required=True, help='Genie json configuration file')
    parser.add_argument('--output_csv', required=True, help='Resulting filtered rows csv file')
    args = parser.parse_args()

    config = load_config(args.genie_config)
    target = config['target_variable']

    # Exit early if target is not in valid
    if target not in outputs.keys():
        sys.exit("WRONG TARGET VARIABLE")

    dataframe: pd.DataFrame = load_dataset(config)

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
            print("no")
        case 'STAGE_EL':
            print("no")

    dataframe[columns].to_csv(args.output_csv, index=False)

    # if file != 'Lower Grade Glioma and Glioblastoma (GBMLGG)':
    #     dataframe = prepare_stage(dataframe)
    #     dataframe["early"] = dataframe.apply(lambda row: 1 if row['Stage 1'] == 1 or row['Stage 2'] == 1 else 0, axis=1)
    #     dataframe["late"] = dataframe.apply(lambda row: 1 if row['Stage 4'] == 1 or row['Stage 3'] == 1 else 0, axis=1)
