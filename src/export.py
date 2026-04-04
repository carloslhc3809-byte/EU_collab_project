import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from .config import DataConfig, ModelConfig
from .utils import to_python


def save_pipeline_outputs(
    artifacts: Dict,
    applicant_to_id: Dict[str, int],
    cfg: DataConfig,
    model_cfg: ModelConfig,
) -> None:
    cfg.ensure_dirs()

    metrics = to_python(artifacts["metrics"])
    with open(cfg.predictions_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    id_to_app = {i: app for app, i in applicant_to_id.items()}
    df_nodes = pd.DataFrame({
        "node_id": list(id_to_app.keys()),
        "applicant_name": list(id_to_app.values()),
    }).sort_values("node_id")

    z = artifacts["z"]
    z_norm = artifacts["z_norm"]
    h_train = artifacts["h_train"]
    train_edge_index = artifacts["train_edge_index"]
    test_edge_index = artifacts["test_edge_index"]
    model = artifacts["model"]

    kmeans = KMeans(n_clusters=model_cfg.k_clusters, random_state=model_cfg.random_seed, n_init=10)
    clusters = kmeans.fit_predict(z.numpy())
    df_nodes["cluster"] = clusters
    df_nodes.to_csv(cfg.metadata_dir / "nodes_with_clusters.csv", index=False)

    np.save(cfg.embeddings_dir / "z_2024.npy", z.numpy())
    np.save(cfg.embeddings_dir / "z_norm.npy", z_norm.numpy())

    temporal_clusters = [
        KMeans(
            n_clusters=model_cfg.k_clusters,
            random_state=model_cfg.random_seed,
            n_init=10,
        ).fit_predict(h_train[t].cpu().numpy())
        for t in range(h_train.shape[0])
    ]
    np.save(cfg.embeddings_dir / "temporal_clusters.npy", np.array(temporal_clusters))

    cluster_matrix = np.zeros((model_cfg.k_clusters, model_cfg.k_clusters), dtype=np.float32)
    for i in range(model_cfg.k_clusters):
        for j in range(model_cfg.k_clusters):
            mask_i = clusters == i
            mask_j = clusters == j
            if mask_i.sum() > 0 and mask_j.sum() > 0:
                sim_block = (z_norm[mask_i] @ z_norm[mask_j].T).numpy()
                cluster_matrix[i, j] = sim_block.mean()
    np.save(cfg.predictions_dir / "cluster_matrix.npy", cluster_matrix)

    np.save(cfg.edges_dir / "train_edges.npy", train_edge_index.numpy())
    np.save(cfg.edges_dir / "test_edges.npy", test_edge_index.numpy())

    known_edges = set()
    for src, dst in train_edge_index.t().tolist():
        a, b = (src, dst) if src < dst else (dst, src)
        known_edges.add((a, b))
    for src, dst in test_edge_index.t().tolist():
        a, b = (src, dst) if src < dst else (dst, src)
        known_edges.add((a, b))

    top_links = {}
    num_nodes = z.size(0)
    model.eval()
    with torch.no_grad():
        for i in range(num_nodes):
            all_targets = torch.arange(num_nodes, dtype=torch.long)
            mask = all_targets != i
            targets = all_targets[mask]
            pair_index = torch.stack([torch.full_like(targets, i), targets], dim=0)
            scores = model.decode_proba(z, pair_index).view(-1)

            scored = []
            for tgt, score in zip(targets.tolist(), scores.tolist()):
                a, b = (i, tgt) if i < tgt else (tgt, i)
                if (a, b) in known_edges:
                    continue
                scored.append({"target": int(tgt), "score": float(score)})

            scored = sorted(scored, key=lambda x: x["score"], reverse=True)[: model_cfg.top_k_predictions]
            top_links[str(i)] = scored

    with open(cfg.predictions_dir / "top_links.json", "w") as f:
        json.dump(top_links, f)

    prob_matrix = torch.zeros((num_nodes, num_nodes))
    with torch.no_grad():
        for i in range(num_nodes):
            all_targets = torch.arange(num_nodes, dtype=torch.long)
            pair_index = torch.stack([torch.full_like(all_targets, i), all_targets], dim=0)
            scores = model.decode_proba(z, pair_index).view(-1)
            prob_matrix[i] = scores.cpu()
    np.save(cfg.predictions_dir / "prob_matrix.npy", prob_matrix.numpy())
