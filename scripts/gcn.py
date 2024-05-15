import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torcheval.metrics.functional import multiclass_f1_score


class GraphConvolutionalNetwork(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, num_nodes, hidden_channels, adj_matrix):
        super(GraphConvolutionalNetwork, self).__init__()
        self.adj_matrix = adj_matrix
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)
        self.a = torch.nn.ReLU()
        self.num_classes = num_classes
        self.num_nodes = num_nodes

    def forward(self, batch):
        X = batch
        # NE DAO BOG NIKOM
        if isinstance(batch, list):
            X = torch.stack(tuple(data.x for data in batch)).reshape(shape=(len(batch), self.num_nodes)).unsqueeze(-1)

        edge_index = torch.concat(
            tuple(torch.nonzero(torch.tensor(self.adj_matrix), as_tuple=False).t().contiguous() for _ in
                  range(len(batch))),
            dim=1)
        x = self.a(self.conv1(X, edge_index))
        x = x.view(len(batch), -1, x.size(-1)).mean(dim=1)
        x = self.lin(x)

        return x


class GCNModelTrainer:
    def __init__(self, model, optimizer, criterion, num_classes, epochs):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.criterion = criterion
        self.num_classes = num_classes

    def train(self, train_loader, test_loader, on_epoch_finished):

        epoch_metrics = []
        epoch_losses = []

        for epoch in range(self.epochs):
            self.model.train()
            tr_metrics = []
            tr_losses = []
            for batch in train_loader:
                self.optimizer.zero_grad()
                out = self.model(batch)
                labels = torch.stack(tuple(data.y for data in batch), dim=1).squeeze(dim=0)
                loss = self.criterion(out, labels)
                tr_losses.append(loss.item())

                out_prob = F.softmax(out, dim=1)
                pred = out_prob.argmax(dim=1)
                Y = torch.stack(tuple(data.y for data in batch), dim=1).squeeze(dim=0)
                _, true_labels = Y.max(dim=1)
                tr_metrics.append(
                    multiclass_f1_score(pred.detach(), torch.Tensor(true_labels), num_classes=self.num_classes))

                loss.backward()
                self.optimizer.step()

            self.model.eval()

            metrics = []
            losses = []
            for batch in test_loader:
                out = self.model(batch)
                out_prob = F.softmax(out, dim=1)
                Y = torch.stack(tuple(data.y for data in batch), dim=1).squeeze(dim=0)
                loss = self.criterion(out, Y)
                losses.append(loss.item())
                pred = out_prob.argmax(dim=1)
                _, true_labels = Y.max(dim=1)
                metrics.append(
                    multiclass_f1_score(pred.detach(), torch.Tensor(true_labels), num_classes=self.num_classes))

            epoch_metrics.append(np.mean(metrics))
            epoch_losses.append(np.mean(losses))
            on_epoch_finished(epoch, np.mean(tr_metrics), np.mean(tr_losses), epoch_metrics[-1], epoch_losses[-1])

        return np.mean(epoch_metrics)
