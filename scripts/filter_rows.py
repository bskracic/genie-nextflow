import argparse
import sys

import pandas as pd


def transform_id(value: str):
    parts = value.split("-")
    if len(parts) < 4:
        return '-'.join(parts)
    else:
        return '-'.join(parts[:-1])


def load_dataset(file: str, clean: str, genes: list) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(file, sep='\t', header=None)
    print('loaded dataset: ', file)
    df = df.T
    df.columns = df.iloc[0]
    df = df[1:]
    df = df.rename(columns={'sample': 'patient_id'})
    df.set_index(pd.Index(range(len(df))), inplace=True)
    df = df.dropna()

    columns = ['patient_id'] + genes
    temp_df = df[columns]

    temp_df[genes] = temp_df[genes].astype(float)

    print('prepared dataset: ', file)

    # transform patient ids so that we can match the same one and take a mean
    temp_df.loc[:, 'patient_id'] = temp_df['patient_id'].apply(transform_id)

    aggregated_df = temp_df.groupby('patient_id').agg(['mean'])
    aggregated_df = aggregated_df.reset_index()
    aggregated_df.columns = temp_df.columns

    survival_dataframe = pd.read_csv(clean, sep='\t')
    survival_dataframe = survival_dataframe.drop(survival_dataframe.columns[0], axis=1)
    survival_dataframe = survival_dataframe.rename(columns={'bcr_patient_barcode': 'patient_id'})

    print(f'patients with gene expression data: {len(aggregated_df)}')
    print(f'patients with survival data: {len(survival_dataframe)}')

    df = aggregated_df.merge(survival_dataframe, on='patient_id', how='inner')

    print(f'patients when merged: {len(df)}')

    df.drop(columns=['patient_id'], inplace=True)
    return df


outputs: dict = {
    'STAGE': ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'],
    'STAGE_EL': ['early', 'late'],
    'DSS': ['DSS_1', 'DSS_0'],
    'OS': ['OS_1', 'OS_0'],
    'GENDER': ['gender_female', 'gender_male']
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter rows by cancer type')
    parser.add_argument('--input_csv', required=True, help='Input CSV file')
    parser.add_argument('--clean_data', required=True, help='Cleaned data containing target variables to join')
    parser.add_argument('--output_csv', required=True, help='Output CSV file')
    parser.add_argument('--cancer_type', required=True, help='Desired cancer type')
    parser.add_argument("--target", required=True, help='Target variable to predict')
    args = parser.parse_args()

    genes = [
        'TREX1',
        'ATM',
        'ENPP1',
        'C6orf150',
        'CCL5',
        'CXCL10',
        'TMEM173',
        'CXCL9',
        'CXCL11',
        'TBK1',
        'IKBKE',
        'IRF3',
        'IRF7',
        'IFNA1',
        'IFNB1',
        'NFKB1'
    ]

    # Exit early if target is not in valid
    if args.target not in outputs.keys():
        sys.exit("WRONG TARGET VARIABLE")

    dataframe: pd.DataFrame = load_dataset(file=args.input_csv, clean=args.clean_data, genes=genes)

    columns = genes
    match args.target:
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
