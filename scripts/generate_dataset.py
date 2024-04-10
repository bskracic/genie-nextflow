import argparse
import pickle

import pandas as pd
import torch
from torch_geometric.data import Data

from genie_utils import load_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate graph dataset')
    parser.add_argument('--genie_config', required=True, help='Genie json configuration file')
    parser.add_argument('--input_csv', required=True, help='Input CSV file')
    parser.add_argument('--dataset_obj', required=True, help='Output pickle object file')
    args = parser.parse_args()

    config = load_config(args.genie_config)
    genes = config['genes']
    num_genes = len(genes)

    df = pd.read_csv(args.input_csv)

    dataset = []
    for _, row in df.iterrows():
        gene_data = torch.tensor(row[:num_genes].values, dtype=torch.float).view(1, -1)
        stage_label = torch.tensor(row[num_genes:].values, dtype=torch.float).view(-1, len(row[num_genes:]))

        dataset.append(Data(x=gene_data, y=stage_label))

    with open(args.dataset_obj, 'wb') as fh:
        pickle.dump(dataset, fh)
