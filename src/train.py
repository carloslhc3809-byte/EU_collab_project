from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.utils import add_self_loops, negative_sampling

from .config import ModelConfig
from .model import SnapshotTemporalGNN


def canonicalize_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index is None or edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)

    arr = edge_index.t().cpu().numpy().astype(np.int64)
    arr = arr[arr[:, 0] != arr[:, 1]]
    if len(arr) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    arr = np.sort(arr, axis=1)
    arr = np.unique(arr, axis=0)
    if len(arr) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(arr, dtype=torch.long).t().contiguous()


def to_undirected_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index is None or edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    rev = edge_index[[1, 0], :]
    out = torch.cat([edge_index, rev], dim=1)
    return canonicalize_edge_index(out)


def get_edge_index_from_store(data: HeteroData, edge_type):
    if edge_type not in data.edge_types:
        return None
    store = data[edge_type]
    if "edge_index" in store and store["edge_index"] is not None:
        return store["edge_index"].long()
    if "edge_label_index" in store and store["edge_label_index"] is not None:
        return store["edge_label_index"].long()
    if "adj_t" in store and store["adj_t"] is not None:
        row, col, _ = store["adj_t"].t().coo()
        return torch.stack([row, col], dim=0).long()
    return None


def find_applicant_patent_edge_types(data: HeteroData):
    files_type = None
    rev_files_type = None
    preferred_forward, preferred_reverse = [], []
    fallback_forward, fallback_reverse = [], []

    for et in data.edge_types:
        src, rel, dst = et
        rel_l = str(rel).lower()

        if src == "applicant" and dst == "patent":
            fallback_forward.append(et)
            if "file" in rel_l:
                preferred_forward.append(et)

        if src == "patent" and dst == "applicant":
            fallback_reverse.append(et)
            if "file" in rel_l:
                preferred_reverse.append(et)

    files_type = preferred_forward[0] if preferred_forward else (fallback_forward[0] if fallback_forward else None)
    rev_files_type = preferred_reverse[0] if preferred_reverse else (fallback_reverse[0] if fallback_reverse else None)
    return files_type, rev_files_type


def infer_year_app_map(year: int, data: HeteroData, global_apps: Dict[str, int]):
    num_local = int(data["applicant"].num_nodes)
    candidate_keys = ["name", "names", "applicant_name", "applicant_names", "label", "labels", "id", "ids"]

    for key in candidate_keys:
        if key in data["applicant"]:
            values = data["applicant"][key]
            values = values.cpu().tolist() if torch.is_tensor(values) else list(values)

            inferred = {}
            ok = True
            for i, v in enumerate(values):
                if isinstance(v, bytes):
                    v = v.decode("utf-8", errors="ignore")
                v = str(v)
                if v not in global_apps:
                    ok = False
                    break
                inferred[v] = i

            if ok and inferred:
                return inferred

    if num_local <= len(global_apps):
        return {i: i for i in range(num_local)}

    raise ValueError(
        f"Could not infer applicant mapping for year {year}. "
        f"Local applicant nodes: {num_local}, global applicants: {len(global_apps)}"
    )


def map_local_applicant_id_to_global(local_id, app_map, global_map):
    if all(isinstance(k, (int, np.integer)) for k in app_map.keys()):
        return app_map.get(local_id, None)

    inv_app_map = {v: k for k, v in app_map.items()}
    applicant_key = inv_app_map.get(local_id, None)
    if applicant_key is None:
        return None
    return global_map.get(applicant_key, None)


def reconstruct_global_collab_edges_from_patents(year: int, data: HeteroData, app_map, global_map):
    files_type, rev_files_type = find_applicant_patent_edge_types(data)
    if files_type is None and rev_files_type is None:
        raise ValueError(
            f"Could not find applicant-patent relations for year {year}. Available edge types: {data.edge_types}"
        )

    app_pat_parts = []
    if files_type is not None:
        ei = get_edge_index_from_store(data, files_type)
        if ei is not None and ei.numel() > 0:
            app_pat_parts.append(ei)

    if rev_files_type is not None:
        ei = get_edge_index_from_store(data, rev_files_type)
        if ei is not None and ei.numel() > 0:
            app_pat_parts.append(torch.stack([ei[1], ei[0]], dim=0))

    if len(app_pat_parts) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    app_pat = torch.cat(app_pat_parts, dim=1)
    app_pat = torch.unique(app_pat.t(), dim=0).t().contiguous()

    patent_to_global_apps = {}
    for local_app_id, patent_id in app_pat.t().tolist():
        global_app_id = map_local_applicant_id_to_global(local_app_id, app_map, global_map)
        if global_app_id is None:
            continue
        if global_app_id < 0 or global_app_id >= len(global_map):
            continue
        patent_to_global_apps.setdefault(patent_id, set()).add(global_app_id)

    collab_edges = set()
    for _, app_set in patent_to_global_apps.items():
        apps = sorted(app_set)
        if len(apps) < 2:
            continue
        for i in range(len(apps)):
            for j in range(i + 1, len(apps)):
                a, b = apps[i], apps[j]
                if a != b:
                    x, y = (a, b) if a < b else (b, a)
                    collab_edges.add((x, y))

    if not collab_edges:
        return torch.empty((2, 0), dtype=torch.long)

    arr = np.array(list(collab_edges), dtype=np.int64)
    return torch.tensor(arr, dtype=torch.long).t().contiguous()


def build_yearly_global_graphs(data_by_year: Dict[int, HeteroData], years: List[int], global_apps: Dict[str, int]):
    yearly_edges = {}
    for year in years:
        year_app_map = infer_year_app_map(year, data_by_year[year], global_apps)
        edge_index = reconstruct_global_collab_edges_from_patents(
            year=year,
            data=data_by_year[year],
            app_map=year_app_map,
            global_map=global_apps,
        )
        edge_index = canonicalize_edge_index(edge_index)
        edge_index = to_undirected_edge_index(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=len(global_apps))
        yearly_edges[year] = edge_index.long()
    return yearly_edges


def normalize_feature_tensor(X: torch.Tensor, train_T: int) -> torch.Tensor:
    X = torch.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0).float()
    X_train_window = X[:train_T]
    mean = X_train_window.mean(dim=(0, 1), keepdim=True)
    std = X_train_window.std(dim=(0, 1), keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    X = (X - mean) / std
    X = torch.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)
    return X.detach().clone()


def train_temporal_link_predictor(
    X: torch.Tensor,
    data_by_year: Dict[int, HeteroData],
    applicant_to_id: Dict[str, int],
    device: torch.device,
    cfg: ModelConfig,
):
    years = sorted(data_by_year.keys())
    if len(years) < 2:
        raise ValueError("Need at least two years for temporal split.")

    train_years = years[:-1]
    test_year = years[-1]
    train_T = len(train_years)

    X = normalize_feature_tensor(X, train_T=train_T)
    X_train = X[:train_T].detach().clone().to(device)

    all_yearly_edges = build_yearly_global_graphs(data_by_year, years, applicant_to_id)
    train_edge_index_seq = [all_yearly_edges[y].to(device) for y in train_years]

    train_edge_list = []
    for y in train_years:
        e = canonicalize_edge_index(all_yearly_edges[y].cpu())
        if e.numel() > 0:
            train_edge_list.append(e)
    if len(train_edge_list) == 0:
        raise ValueError("No training collaboration edges were reconstructed from train years.")

    train_edge_index = canonicalize_edge_index(torch.cat(train_edge_list, dim=1)).to(device)
    if train_edge_index.numel() == 0:
        raise ValueError("Training edge index is empty after cleaning.")

    test_edge_index = canonicalize_edge_index(all_yearly_edges[test_year].cpu()).to(device)
    if test_edge_index.numel() == 0:
        raise ValueError("No test collaboration edges were reconstructed for the final year.")

    model = SnapshotTemporalGNN(
        in_dim=X_train.shape[-1],
        gnn_hidden_dim=cfg.gnn_hidden_dim,
        rnn_hidden_dim=cfg.rnn_hidden_dim,
        gnn_layers=cfg.gnn_layers,
        dropout=cfg.dropout,
        use_attention=cfg.use_attention,
        predictor=cfg.predictor,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    losses, auc_history, ap_history = [], [], []

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        _, z_train = model.encode(X_train.detach(), train_edge_index_seq)
        pos_logits = model.decode_logits(z_train, train_edge_index)

        neg_edge_index = negative_sampling(
            edge_index=train_edge_index,
            num_nodes=z_train.size(0),
            num_neg_samples=train_edge_index.size(1),
            method="sparse",
        ).to(device)
        neg_logits = model.decode_logits(z_train, neg_edge_index)

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)

        emb_penalty = cfg.emb_penalty_weight * z_train.pow(2).mean()
        loss = criterion(logits, labels) + emb_penalty

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            model.eval()
            with torch.no_grad():
                _, z_eval = model.encode(X_train.detach(), train_edge_index_seq)
                pos_pred = model.decode_proba(z_eval, test_edge_index, clip=cfg.sigmoid_clip)

                neg_test_edge_index = negative_sampling(
                    edge_index=test_edge_index,
                    num_nodes=z_eval.size(0),
                    num_neg_samples=test_edge_index.size(1),
                    method="sparse",
                ).to(device)
                neg_pred = model.decode_proba(z_eval, neg_test_edge_index, clip=cfg.sigmoid_clip)

                preds = torch.cat([pos_pred, neg_pred]).cpu().numpy()
                labels_eval = torch.cat([
                    torch.ones(pos_pred.size(0), device=device),
                    torch.zeros(neg_pred.size(0), device=device),
                ]).cpu().numpy()

                auc = roc_auc_score(labels_eval, preds)
                ap = average_precision_score(labels_eval, preds)
                auc_history.append((epoch, float(auc)))
                ap_history.append((epoch, float(ap)))

    model.eval()
    with torch.no_grad():
        h_train, z_train = model.encode(X_train.detach(), train_edge_index_seq)
        pos_pred = model.decode_proba(z_train, test_edge_index, clip=cfg.sigmoid_clip)

        neg_test_edge_index = negative_sampling(
            edge_index=test_edge_index,
            num_nodes=z_train.size(0),
            num_neg_samples=test_edge_index.size(1),
            method="sparse",
        ).to(device)
        neg_pred = model.decode_proba(z_train, neg_test_edge_index, clip=cfg.sigmoid_clip)

        preds = torch.cat([pos_pred, neg_pred]).cpu().numpy()
        labels_eval = torch.cat([
            torch.ones(pos_pred.size(0), device=device),
            torch.zeros(neg_pred.size(0), device=device),
        ]).cpu().numpy()

        auc = roc_auc_score(labels_eval, preds)
        ap = average_precision_score(labels_eval, preds)

    z = z_train.detach().cpu()
    z_norm = z / (z.norm(dim=1, keepdim=True) + 1e-8)

    metrics = {
        "losses": losses,
        "auc_history": [{"epoch": int(e), "auc": float(v)} for e, v in auc_history],
        "ap_history": [{"epoch": int(e), "ap": float(v)} for e, v in ap_history],
        "final_auc": float(auc),
        "final_ap": float(ap),
        "train_years": [int(y) for y in train_years],
        "test_year": int(test_year),
        "num_train_edges": int(train_edge_index.size(1)),
        "num_test_edges": int(test_edge_index.size(1)),
        "score_diagnostics": {
            "positive_max": float(pos_pred.max().item()),
            "positive_mean": float(pos_pred.mean().item()),
            "negative_max": float(neg_pred.max().item()),
            "negative_mean": float(neg_pred.mean().item()),
        },
        "model_config": {
            "predictor": cfg.predictor,
            "temporal": "attention" if cfg.use_attention else "mean",
            "dropout": cfg.dropout,
            "activation": "LeakyReLU",
            "hidden_dim": cfg.rnn_hidden_dim,
        },
    }

    artifacts = {
        "model": model,
        "metrics": metrics,
        "z": z,
        "z_norm": z_norm,
        "h_train": h_train.detach().cpu(),
        "train_edge_index": train_edge_index.detach().cpu(),
        "test_edge_index": test_edge_index.detach().cpu(),
        "train_years": train_years,
        "test_year": test_year,
    }
    return artifacts
