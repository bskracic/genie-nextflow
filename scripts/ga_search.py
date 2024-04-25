import argparse
from datetime import datetime
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pygad
import torch
from torch_geometric.loader import DataListLoader
import wandb

from gcn import GraphConvolutionalNetwork, GCNModelTrainer
from genie_utils import load_config

LR = 0.0001
WD = 1e-1
HIDDEN_SIZE = 64
NUM_SOLUTIONS = 4
NUM_GENERATIONS = 1
NUM_PARENTS_MATING = 2
criterion = torch.nn.CrossEntropyLoss()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find ')
    parser.add_argument('--genie_config', required=True, help='Genie json configuration file')
    parser.add_argument('--dataset_obj', required=True, help='Input pickle object file')
    parser.add_argument('--cancers', required=True, help='Comma separated list of cancers')
    parser.add_argument("--adj_matrix_obj", required=True, help='Output adjacency matrix pickle')
    parser.add_argument('--target', required=True, help='Target variable')
    parser.add_argument('--wandb_api_key', required=True, help='wandb secret API key')

    args = parser.parse_args()

    with open(args.dataset_obj, 'rb') as fh:
        dataset = pickle.load(fh)

    split_idx = round(0.75 * len(dataset))
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]

    train_loader = DataListLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataListLoader(test_dataset, batch_size=64, shuffle=True)

    NODES = dataset[0].x.shape[-1]
    ADJ_MATRIX_SHAPE = (NODES, NODES)
    NUM_CLASSES = dataset[0].y.shape[-1]

    config = load_config(args.genie_config)
    # Initialize wandb
    wandb.login(key=args.wandb_api_key)
    run = wandb.init(
        project="GENIE-Nextflow",
        name=f'{str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))}_{args.cancers}' + f"_target_{args.target}",
        config={
            "cancers": args.cancers,
            'variable': args.target,
            "learning_rate": LR,
            "wd": WD,
            "hidden_size": HIDDEN_SIZE,
            "architecture": "GCN",
            "nodes": NODES,
            "genes": config['genes'],
            "epochs": 10,
        }
    )


    def epoch_finished(a, b, c):
        # print('#', end='')
        pass


    def fitness_function(ga: pygad.GA, solution, index):
        adj_matrix = solution.reshape(ADJ_MATRIX_SHAPE)
        model = GraphConvolutionalNetwork(num_node_features=1, num_classes=NUM_CLASSES, hidden_channels=HIDDEN_SIZE,
                                          adj_matrix=adj_matrix)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        trainer = GCNModelTrainer(model=model, optimizer=optimizer, criterion=criterion,
                                  num_classes=NUM_CLASSES, epochs=10)

        final_f1s = trainer.train(train_loader=train_loader, test_loader=test_loader, on_epoch_finished=epoch_finished)
        print(' =>', np.mean(final_f1s))
        return np.mean(final_f1s)


    def generation_finished(ga: pygad.GA):
        solution, fitness, _ = ga.best_solution()
        print('Generation fitness:', fitness)

    starting_flat_matrix = np.array(np.random.randint(2, size=(NODES, NODES)), dtype=float).flatten()
    ga_instance = pygad.GA(
        num_generations=NUM_GENERATIONS,
        mutation_probability=0.7,
        num_parents_mating=NUM_PARENTS_MATING,
        sol_per_pop=NUM_SOLUTIONS,
        num_genes=starting_flat_matrix.shape,
        fitness_func=fitness_function,
        gene_space=[0, 1],
        initial_population=[starting_flat_matrix.copy() for _ in range(NUM_SOLUTIONS)],
        on_generation=generation_finished,
    )

    def draw_graph(adj_matrix) -> plt.Figure:
        fig = plt.figure(figsize=(10, 8))
        G = nx.from_numpy_array(adj_matrix)
        labels = {i: gene for i, gene in enumerate(config['genes'])}
        pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos, labels=labels, with_labels=True, node_color='skyblue', node_size=500, font_size=12)
        return fig


    print('started GA search...')
    ga_instance.run()

    best_solution, best_solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=best_solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=best_solution_fitness))
    print("Parameters of the best solution : {solution}".format(solution=best_solution))
    print(
        "Fitness value of the best solution = {solution_fitness}".format(solution_fitness=best_solution_fitness))

    adj_matrix = best_solution.reshape(NODES, NODES)

    run.log(
        {"adjacency_matrix": adj_matrix,
         "graph_figure": wandb.Image(draw_graph(adj_matrix=adj_matrix)),
         'best_f1': best_solution_fitness})

    with open(args.adj_matrix_obj, 'wb') as fh:
        pickle.dump(adj_matrix, fh)

    with open("wandb_run_id.txt", "w") as f:
        f.write(run.id)

