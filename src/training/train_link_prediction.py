import torch
import random
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_features(X, train_T):
    mean = X[:train_T].mean()
    std = X[:train_T].std() + 1e-6
    X = (X - mean) / std
    return X, X[:train_T]


def build_global_edge_index(local_edge_index, app_map, global_map):

    inv_app_map = {v: k for k, v in app_map.items()}
    edges = []

    for src, dst in local_edge_index.t().tolist():

        if src not in inv_app_map:
            continue

        src_name = inv_app_map[src]

        if src_name not in global_map:
            continue

        src_global = global_map[src_name]
        dst_global = global_map[src_name] if src_name in global_map else None

        if dst_global is None:
            continue

        edges.append((src_global, dst_global))

    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_train_test_edges(data_by_year, app_maps, global_apps, train_ratio=0.8):

    years = sorted(data_by_year.keys())
    train_T = int(len(years) * train_ratio)

    global_map = {app: i for i, app in enumerate(global_apps)}

    train_edges = []
    test_edges = []

    for t, year in enumerate(years):

        data = data_by_year[year]
        app_map = app_maps[year]

        if ("applicant", "files", "patent") not in data.edge_types:
            continue

        local_edges = data[("applicant", "files", "patent")].edge_index

        edge_index = build_global_edge_index(
            local_edges,
            app_map,
            global_map
        )

        if edge_index.numel() == 0:
            continue

        if t < train_T:
            train_edges.append(edge_index)
        else:
            test_edges.append(edge_index)

    train_edge_index = torch.cat(train_edges, dim=1)
    test_edge_index = torch.cat(test_edges, dim=1)

    return train_edge_index, test_edge_index, train_T


def train(model, X, train_edge_index, epochs=70, lr=0.01):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):

        model.train()

        pred, z = model(X, train_edge_index)

        pos_pred = pred

        neg_edge_index = negative_sampling(
            train_edge_index,
            num_nodes=z.size(0),
            num_neg_samples=train_edge_index.size(1)
        )

        neg_pred = model.predictor(z, neg_edge_index)

        eps = 1e-15

        loss = -torch.log(pos_pred + eps).mean() \
               -torch.log(1 - neg_pred + eps).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model


def evaluate(model, X, test_edge_index):

    model.eval()

    with torch.no_grad():
        pred, z = model(X, test_edge_index)

    pos_pred = pred.cpu().numpy()

    neg_edge_index = negative_sampling(
        test_edge_index,
        num_nodes=z.size(0),
        num_neg_samples=test_edge_index.size(1)
    )

    neg_pred = model.predictor(z, neg_edge_index).cpu().numpy()

    y_true = np.concatenate([
        np.ones_like(pos_pred),
        np.zeros_like(neg_pred)
    ])

    y_scores = np.concatenate([pos_pred, neg_pred])

    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    return auc, ap
