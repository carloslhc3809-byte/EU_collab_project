from typing import Dict, Tuple

import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import MetaPath2Vec

from .config import FeatureConfig


def train_metapath_embedding(
    data: HeteroData,
    metapath,
    device: torch.device,
    cfg: FeatureConfig,
) -> torch.Tensor:
    num_nodes = data["applicant"].num_nodes
    try:
        model = MetaPath2Vec(
            data.edge_index_dict,
            embedding_dim=cfg.emb_dim,
            metapath=metapath,
            walk_length=cfg.walk_length,
            context_size=cfg.context_size,
            walks_per_node=cfg.walks_per_node,
            num_negative_samples=cfg.num_negative_samples,
            sparse=True,
        ).to(device)

        optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
        loader = model.loader(batch_size=cfg.batch_size, shuffle=True)

        model.train()
        for _ in range(cfg.epochs):
            for pos_rw, neg_rw in loader:
                pos_rw = pos_rw.to(device)
                neg_rw = neg_rw.to(device)
                optimizer.zero_grad()
                loss = model.loss(pos_rw, neg_rw)
                if torch.isnan(loss):
                    continue
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            emb = model("applicant").cpu()
        emb = torch.nan_to_num(emb, nan=0.0, posinf=5.0, neginf=-5.0)
    except Exception:
        emb = torch.zeros(num_nodes, cfg.emb_dim)

    if emb.shape[0] != num_nodes:
        full_emb = torch.zeros(num_nodes, cfg.emb_dim)
        full_emb[: emb.shape[0]] = emb
        emb = full_emb

    return emb


def normalize_embeddings(z: torch.Tensor) -> torch.Tensor:
    return z / (z.norm(dim=1, keepdim=True) + 1e-8)


def generate_temporal_embeddings(
    data_by_year: Dict[int, HeteroData],
    device: torch.device,
    cfg: FeatureConfig,
) -> Dict[int, torch.Tensor]:
    emb_by_year = {}
    for year, data in data_by_year.items():
        emb_list = []
        for _, meta_cfg in cfg.metapaths.items():
            emb = train_metapath_embedding(data, meta_cfg["path"], device=device, cfg=cfg)
            emb = emb * meta_cfg["weight"]
            emb_list.append(emb)
        z = torch.cat(emb_list, dim=1)
        z = normalize_embeddings(z)
        emb_by_year[year] = z
    return emb_by_year


def build_temporal_tensor(
    emb_by_year: Dict[int, torch.Tensor],
    applicant_to_id: Dict[str, int],
) -> Tuple[torch.Tensor, list]:
    years_sorted = sorted(emb_by_year.keys())
    num_years = len(years_sorted)
    num_nodes = len(applicant_to_id)
    emb_dim = next(iter(emb_by_year.values())).shape[1]

    X = torch.zeros(num_years, num_nodes, emb_dim)
    for t, year in enumerate(years_sorted):
        X[t] = emb_by_year[year]
    return X, years_sorted
