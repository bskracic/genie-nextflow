import argparse
import math
import pickle
from tqdm import tqdm

from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataListLoader
import wandb

from genie_utils import load_config
from gcn import GraphConvolutionalNetwork, GCNModelTrainer

outputs: dict = {
    'STAGE': ['early', 'late'],
    'DSS': ['DSS_1', 'DSS_0'],
    'OS': ['OS_1', 'OS_0'],
    'GENDER': ['gender_female', 'gender_male']
}

LR = 0.0001
WD = 1e-1
HIDDEN_SIZE = 64
EPOCHS = 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute IG attributions for each problem')
    parser.add_argument('--genie_config', required=True, help='Genie json configuration file')
    parser.add_argument('--wandb_api_key', required=True, help='W&B API key')
    parser.add_argument('--cancers', required=True, help='List of cancers')
    parser.add_argument('--target', required=True, help='Target variable')
    parser.add_argument('--dataset_obj', required=True, help='Input pickle object file')
    parser.add_argument('--input_csv', required=True, help='Input dataset csv file')
    parser.add_argument('--adj_matrix', required=True, help='Input adjacency matrix object file')
    # parser.add_argument('--wandb_run_id', required=True, help='W&B run id to resume')
    args = parser.parse_args()

    # with open("wandb_run_id.txt", "r") as f:
    #     run_id = f.read().strip()
    # wandb.login(key=args.wandb_api_key)
    # run = wandb.init(id=run_id, resume="must", project='GENIE-Nextflow-v3')

    with open(args.dataset_obj, 'rb') as fh:
        dataset = pickle.load(fh)

    with open(args.adj_matrix, 'rb') as fh:
        adj_matrix: list[list[int]] = pickle.load(fh)

    target = args.target
    cancer = args.cancers
    NUM_GENES = 13
    config = load_config(args.genie_config)
    list_of_genes = config['genes']
    NODES = dataset[0].x.shape[-1]
    ADJ_MATRIX_SHAPE = (NODES, NODES)
    NUM_CLASSES = dataset[0].y.shape[-1]

    split_idx = round(0.75 * len(dataset))
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]

    train_loader = DataListLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataListLoader(test_dataset, batch_size=64, shuffle=True)

    device = 'cpu'
    if torch.cuda.is_available():
        print(f'CUDA found, using {torch.cuda.get_device_name("cuda:0")}')
        device = 'cuda:0'

    model = GraphConvolutionalNetwork(num_node_features=1, num_nodes=NODES, num_classes=NUM_CLASSES,
                                      hidden_channels=HIDDEN_SIZE,
                                      adj_matrix=adj_matrix, device=device).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    trainer = GCNModelTrainer(model=model, optimizer=optimizer, criterion=criterion,
                              num_classes=NUM_CLASSES, epochs=EPOCHS)
    print('training started')
    pbar = tqdm(total=EPOCHS, desc="Training Progress")

    def epoch_finished(epoch, tr_f1, tr_loss, f1, loss):
        # print({"epoch": epoch, "train_f1": tr_f1, "train_loss": tr_loss, "f1": f1, "loss": loss})
        pbar.update(1)
        pass


    final_f1 = trainer.train(train_loader=train_loader, test_loader=test_loader, on_epoch_finished=epoch_finished)
    print({"f1": final_f1})
    print('\nfinished')
    pbar.close()


    def draw_graph(attrs):
        cmap = plt.cm.RdPu
        colors = [cmap(val) for val in attrs]

        fig = plt.figure(figsize=(10, 8))
        G = nx.from_numpy_array(np.array(adj_matrix))
        labels = {i: gene for i, gene in enumerate(config['genes'])}
        pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos, labels=labels, with_labels=True, node_color=colors, node_size=500, font_size=12)

        return fig


    feature_importance = np.array([])

    df = pd.read_csv(args.input_csv)

    for class_index in range(NUM_CLASSES):

        temp_dataset = []
        _class = outputs[target][class_index]
        for _, row in df[df[_class] == 1].iterrows():
            gene_data = torch.tensor(row[:NUM_GENES].values, dtype=torch.float).view(1, -1)
            label = torch.tensor(row[NUM_GENES:].values, dtype=torch.float).view(-1, len(row[NUM_GENES:]))
            temp_dataset.append(Data(x=gene_data, y=label))

        data_loader = DataListLoader(temp_dataset, batch_size=64)

        total_attributions = []
        for i, sample in tqdm(enumerate(temp_dataset), total=len(temp_dataset), position=0, leave=False):
            integrated_gradients = IntegratedGradients(model)
            attributions = integrated_gradients.attribute(inputs=sample.x.unsqueeze(-1).cuda(), target=class_index)
            attributions = attributions.squeeze().cpu().numpy()
            total_attributions.append(np.abs(attributions))

        feature_importance = np.mean(total_attributions, axis=0)

        print(outputs[target][class_index], ':', feature_importance)
        fig = draw_graph([math.log10(val + 10) for val in feature_importance])
        print(
            {
                "class": outputs[args.target][class_index],
                "attributions": feature_importance.tolist(),
                "feature_importance_graph": wandb.Image(fig, caption=outputs[args.target][class_index])
            })

    # run.finish()
