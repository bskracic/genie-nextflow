import argparse
import pickle

import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find ')
    parser.add_argument('--dataset_obj', required=True, help='Input pickle object file')
    args = parser.parse_args()

    with open(args.dataset_obj, 'rb') as fh:
        df = pickle.load(fh)

    print(df)
