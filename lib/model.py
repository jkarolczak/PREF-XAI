import numpy as np
import torch
import torch.nn as nn

from lib.data import IDataset
from lib.tools import set_seed


class Layer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, batch_norm: bool = False) -> None:
        super().__init__()
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)

        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.activation(x)
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, input_dim: int, cat_idxs: list[int], cat_dims: list[int], embedding_dim: int = 2) -> None:
        super().__init__()
        self.embedding_idxs = [idx for idx, dim in zip(cat_idxs, cat_dims) if dim > 2]
        self.output_dim = input_dim - len(self.embedding_idxs) + len(self.embedding_idxs) * embedding_dim
        self.embedders = nn.ModuleList([nn.Embedding(dim, embedding_dim) for dim in cat_dims if dim > 2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.embedding_idxs) == 0:
            return x.float()
        embeddings = [embedder(x[:, idx].int()).float() for idx, embedder in zip(self.embedding_idxs, self.embedders)]
        embeddings = torch.cat(embeddings, dim=1)
        x_remain = torch.cat([x[:, idx].unsqueeze(1) for idx in range(x.shape[1]) if idx not in self.embedding_idxs], dim=1)
        x = torch.cat([x_remain, embeddings], dim=1).float()
        return x


class MLPWithEmbedding(nn.Module):
    def __init__(self, input_dim: int, cat_idxs: list[int], cat_dims: list[int], hidden_dims: list[int], output_dim: int,
                 embedding_dim: int = 2) -> None:
        super().__init__()
        self.embedder = EmbeddingLayer(input_dim, cat_idxs, cat_dims, embedding_dim)
        self.layers = nn.ModuleList([Layer(input_dim=self.embedder.output_dim, output_dim=hidden_dims[0])])
        for i in range(1, len(hidden_dims)):
            self.layers.append(Layer(input_dim=hidden_dims[i - 1], output_dim=hidden_dims[i]))
        self.layers.append(Layer(input_dim=hidden_dims[-1], output_dim=output_dim))
        self.softmax = nn.Softmax(dim=1)

        for layer in self.layers:
            if hasattr(layer, "fc"):
                nn.init.kaiming_normal_(layer.fc.weight, nonlinearity='relu')
                nn.init.zeros_(layer.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x)
        for layer in self.layers:
            x = layer(x)
        x = self.softmax(x)
        return x


def train(
        model: nn.Module, dataset: IDataset,
        device: torch.device, verbose: bool = False,
        learning_rate: float = 1e-3, weight_decay: float = 1e-5, pos_weight: list[float] = [1.0, 1.0],
        epochs: int = 150, batch_size: int = 1024,
        threshold: float = 0.5
) -> None:
    set_seed()

    train_ds = dataset.train.as_torch(device=device)
    test_ds = dataset.test.as_torch(device=device)

    train_dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*train_ds), batch_size=batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*test_ds), batch_size=batch_size, shuffle=False)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for e in range(epochs):
        if verbose: print(f"Epoch {e + 1}/{epochs}")
        model.train()
        total_loss = 0
        for x, y in train_dl:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)

            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if verbose: print(f"Train loss: {total_loss / len(train_dl)}")
        model.eval()
        with torch.no_grad():
            loss = 0
            preds, targets = [], []
            for x, y in test_dl:
                y_hat = model(x)
                loss += criterion(y_hat, y).item()
                preds.extend(y_hat.cpu().numpy())
                targets.extend(y.cpu().numpy())

            if verbose:
                print(f"Train loss: {loss / len(test_dl)}")

                targets = np.array(targets).argmax(axis=1)
                preds = (np.array(preds)[:, 1] > threshold).astype(int)

                tp = sum((targets == 1) & (preds == 1))
                fp = sum((targets == 0) & (preds == 1))
                fn = sum((targets == 1) & (preds == 0))
                tn = sum((targets == 0) & (preds == 0))

                print(f"Accuracy: {(tp + tn) / len(preds): .4f}")
                print("TP:", tp, "FP:", fp, "\nFN:", fn, "TN:", tn)
                print(f"Recall: {tp / (tp + fn): .4f}", f"Precision: {tp / (tp + fp): .4f}",
                      f"F1: {2 * tp / (2 * tp + fp + fn): .4f}")
