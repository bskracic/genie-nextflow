import argparse
import pickle

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate graph dataset')
    parser.add_argument('--input_csv', required=True, help='Input CSV file')
    parser.add_argument('--dataset_obj', required=True, help='Output pickle object file')
    parser.add_argument("--genes", required=True, help='Genes')
    args = parser.parse_args()

    args.genes = [
        'C6orf150',
        'CCL5',
        'CXCL10',
        'TMEM173',
        'CXCL9',
        'CXCL11',
        'NFKB1',
        'IKBKE',
        'IRF3',
        'TREX1',
        'ATM'
    ]
    num_genes = len(args.genes)

    df = pd.read_csv(args.input_csv)
    scaler = MinMaxScaler()
    df[args.genes] = scaler.fit_transform(df[args.genes])

    dataset = []
    for _, row in df.iterrows():
        gene_data = torch.tensor(row[:num_genes].values, dtype=torch.float).view(1, -1)
        stage_label = torch.tensor(row[num_genes:].values, dtype=torch.float).view(-1, len(row[num_genes:]))

        dataset.append(Data(x=gene_data, y=stage_label))

    with open(args.dataset_obj, 'wb') as fh:
        pickle.dump(dataset, fh)
