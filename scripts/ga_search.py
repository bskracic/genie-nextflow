import argparse
import pickle

import numpy as np
import pygad
import torch
from torch_geometric.loader import DataListLoader

from gcn import GraphConvolutionalNetwork, GCNModelTrainer

LR = 0.0001
WD = 1e-1
HIDDEN_SIZE = 64
criterion = torch.nn.CrossEntropyLoss()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find ')
    parser.add_argument('--dataset_obj', required=True, help='Input pickle object file')
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

    def epoch_finished(a, b, c):
        print('#', end='')

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
        # logger.write(f"Generation {ga.generations_completed}, best_f1: {fitness}\n")
        # logger.write(f"{solution}\n")
        print('\n', solution, '\n', fitness)


    starting_flat_matrix = np.array(np.random.randint(2, size=(NODES, NODES)), dtype=float).flatten()
    num_solutions = 10
    num_generations = 100
    ga_instance = pygad.GA(
        num_generations=num_generations,
        mutation_probability=0.7,
        num_parents_mating=5,
        sol_per_pop=num_solutions,
        num_genes=starting_flat_matrix.shape,
        fitness_func=fitness_function,
        gene_space=[0, 1],
        initial_population=[starting_flat_matrix.copy() for _ in range(num_solutions)],
        on_generation=generation_finished,
    )

    ga_instance.run()

    best_solution, best_solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=best_solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=best_solution_fitness))
    print("Parameters of the best solution : {solution}".format(solution=best_solution))
    print(
        "Fitness value of the best solution = {solution_fitness}".format(solution_fitness=best_solution_fitness))
