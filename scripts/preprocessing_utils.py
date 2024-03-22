import re

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def transform_id(value: str):
    parts = value.split("-")
    if len(parts) < 4:
        return '-'.join(parts)
    else:
        return '-'.join(parts[:-1])


def load_dataset(file: str) -> pd.DataFrame:
    df = pd.read_csv(file, sep='\t', header=None)
    print('loaded dataset: ', file)
    df = df.T
    df.columns = df.iloc[0]
    df = df[1:]
    df = df.rename(columns={'sample': 'patient_id'})
    df.set_index(pd.Index(range(len(df))), inplace=True)
    df = df.dropna()

    temp_df = df[[
        'patient_id',
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
        'NFKB1',
    ]]

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
    temp_df[genes] = temp_df[genes].astype(float)

    print('prepared dataset: ', file)

    # transform patient ids so that we can match the same one and take a mean
    temp_df.loc[:, 'patient_id'] = temp_df['patient_id'].apply(transform_id)

    aggregated_df = temp_df.groupby('patient_id').agg(['mean'])
    aggregated_df = aggregated_df.reset_index()
    aggregated_df.columns = temp_df.columns

    survival_dataframe = pd.read_csv('TCGA_survival_data_clean.txt', sep='\t')
    survival_dataframe = survival_dataframe.drop(survival_dataframe.columns[0], axis=1)
    survival_dataframe = survival_dataframe.rename(columns={'bcr_patient_barcode': 'patient_id'})

    print(f'patients with gene expression data: {len(aggregated_df)}')
    print(f'patients with survival data: {len(survival_dataframe)}')

    df = aggregated_df.merge(survival_dataframe, on='patient_id', how='inner')

    print(f'patients when merged: {len(df)}')

    return df


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

    sample_df = df[[
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
        'NFKB1',
        'ajcc_pathologic_tumor_stage',
        'DSS',
        'OS',
        'gender'
    ]]

    encoder = OneHotEncoder(sparse=False)
    encoded_array = encoder.fit_transform(sample_df[['ajcc_pathologic_tumor_stage']])
    columns = [str.replace(val, 'ajcc_pathologic_tumor_stage_', '') for val in
               encoder.get_feature_names_out(['ajcc_pathologic_tumor_stage'])]
    out_encoded_df = pd.DataFrame(encoded_array, columns=columns)
    sample_df = sample_df.drop(['ajcc_pathologic_tumor_stage'], axis=1)
    sample_df = pd.concat([sample_df, out_encoded_df], axis=1)

    return sample_df