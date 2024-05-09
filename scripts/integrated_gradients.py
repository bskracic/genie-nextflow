import argparse
import pickle

from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.loader import DataListLoader
import wandb

from genie_utils import load_config
from gcn import GraphConvolutionalNetwork, GCNModelTrainer

LR = 0.0001
WD = 1e-1
HIDDEN_SIZE = 64
EPOCHS = 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--genie_config', required=True, help='Genie json configuration file')
    parser.add_argument('--dataset_obj', required=True, help='Input pickle object file')
    parser.add_argument("--adj_matrix_obj", required=True, help='Input adjacency matrix pickle')
    parser.add_argument('--target', required=True, help='Target variable')
    parser.add_argument('--wandb_api_key', required=True, help='wandb secret API key')

    args = parser.parse_args()

    with open(args.dataset_obj, 'rb') as fh:
        dataset = pickle.load(fh)

    with open(args.adj_matrix_obj, 'rb') as fh:
        adj_matrix = pickle.load(fh)

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

    with open("wandb_run_id.txt", "r") as f:
        run_id = f.read().strip()
    wandb.login(key=args.wandb_api_key)
    run = wandb.init(id=run_id, resume="must", project='GENIE-Nextflow')

    def epoch_finished(epoch, tr_f1, tr_loss, f1, loss):
        run.log({"epoch": epoch, "train_f1": tr_f1, "train_loss": tr_loss, "f1": f1, "loss": loss})
        pass

    model = GraphConvolutionalNetwork(num_node_features=1, num_classes=NUM_CLASSES, hidden_channels=HIDDEN_SIZE,
                                      adj_matrix=adj_matrix)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    trainer = GCNModelTrainer(model=model, optimizer=optimizer, criterion=criterion,
                              num_classes=NUM_CLASSES, epochs=EPOCHS)
    print('training started')
    trainer.train(train_loader=train_loader, test_loader=test_loader, on_epoch_finished=epoch_finished)
    print('\nfinished')

    # Maybe refactor this
    outputs: dict = {
        'STAGE': ['early', 'late'],
        'DSS': ['DSS_1', 'DSS_0'],
        'OS': ['OS_1', 'OS_0'],
        'GENDER': ['gender_female', 'gender_male']
    }

    for class_index in range(NUM_CLASSES):
        ig = IntegratedGradients(model)
        first_batch = train_loader.__iter__().__next__()
        attributions, _ = ig.attribute(inputs=first_batch[0].x.unsqueeze(-1), target=class_index,
                                       return_convergence_delta=True)
        feature_importance = attributions.sum(dim=-1).detach().numpy().flatten()

        normalized_data = np.array(feature_importance)
        normalized_data = (normalized_data - np.min(normalized_data)) / (
                np.max(normalized_data) - np.min(normalized_data))

        cmap = plt.cm.RdPu
        colors = [cmap(val) for val in normalized_data]

        fig = plt.figure(figsize=(10, 8))
        G = nx.from_numpy_array(adj_matrix)
        labels = {i: gene for i, gene in enumerate(config['genes'])}
        pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos, labels=labels, with_labels=True, node_color=colors, node_size=500, font_size=12)

        print(outputs[args.target][class_index], ':', feature_importance)

        run.log(
            {
                "class": outputs[args.target][class_index],
                "attributions": str(feature_importance),
                "feature_importance_graph": wandb.Image(fig, caption=outputs[args.target][class_index])
            })

    run.finish()
