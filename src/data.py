import ast
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import torch
from rapidfuzz import fuzz
from torch_geometric.data import HeteroData

from .utils import LIST_COLUMNS, TOKEN_REPLACEMENTS


def resolve_data_path(candidates: List[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    checked = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(f"No dataset found. Checked:\n{checked}")


def parse_list_column(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return [i.strip() for i in str(x).split(";")]


def ensure_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    return [x]


def normalize_applicant_name(name: str) -> str:
    if pd.isna(name):
        return ""
    name = str(name).lower().strip()
    for char in ("&", ",", "."):
        name = name.replace(char, " ")
    name = " ".join(name.split())
    tokens = [TOKEN_REPLACEMENTS.get(tok, tok) for tok in name.split()]
    return " ".join(tokens)


def fuzzy_merge_applicants(applicant_names: Iterable[str], threshold: int = 93):
    applicant_names = sorted(set(a for a in applicant_names if a))
    canonical_map, canonicals, merge_pairs = {}, [], []
    for name in applicant_names:
        for canon in canonicals:
            score = fuzz.ratio(name, canon)
            if score >= threshold:
                canonical_map[name] = canon
                merge_pairs.append((name, canon, score))
                break
        else:
            canonical_map[name] = name
            canonicals.append(name)
    return canonical_map, merge_pairs


def make_mapping(items: Iterable[str]) -> Dict[str, int]:
    return {v: i for i, v in enumerate(sorted(set(items)))}


def to_edge_index(edges: List[List[int]]) -> torch.Tensor:
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def add_edge_type(data: HeteroData, src_type: str, rel: str, dst_type: str, edges: List[List[int]]) -> None:
    edge_index = to_edge_index(edges)
    data[src_type, rel, dst_type].edge_index = edge_index
    reverse_edges = [[j, i] for i, j in edges]
    data[dst_type, f"rev_{rel}", src_type].edge_index = to_edge_index(reverse_edges)


def build_pyg_hetero_graph(
    df: pd.DataFrame,
    applicant_to_id: Dict[str, int],
    inventor_to_id: Dict[str, int],
    patent_to_id: Dict[str, int],
    cpc_to_id: Dict[str, int],
) -> HeteroData:
    data = HeteroData()
    app_pat_edges, inv_pat_edges, cpc_pat_edges = [], [], []

    for _, row in df.iterrows():
        patents = row["Priority Numbers"] or [row["Lens ID"]]
        for patent in patents:
            if patent not in patent_to_id:
                continue
            patent_id = patent_to_id[patent]
            for applicant in row["Applicants"]:
                if applicant in applicant_to_id:
                    app_pat_edges.append([applicant_to_id[applicant], patent_id])
            for inventor in row["Inventors"]:
                if inventor in inventor_to_id:
                    inv_pat_edges.append([inventor_to_id[inventor], patent_id])
            for cpc in row["CPC Classifications"]:
                if cpc in cpc_to_id:
                    cpc_pat_edges.append([cpc_to_id[cpc], patent_id])

    add_edge_type(data, "applicant", "files", "patent", app_pat_edges)
    add_edge_type(data, "inventor", "invented", "patent", inv_pat_edges)
    add_edge_type(data, "cpc", "in_patent", "patent", cpc_pat_edges)

    data["applicant"].num_nodes = len(applicant_to_id)
    data["inventor"].num_nodes = len(inventor_to_id)
    data["patent"].num_nodes = len(patent_to_id)
    data["cpc"].num_nodes = len(cpc_to_id)
    return data


def load_and_prepare_dataset(data_path: Path, fuzzy_threshold: int = 93):
    raw_data = pd.read_csv(data_path)
    df = raw_data[raw_data["Publication Year"] < 2025].copy()

    for col in LIST_COLUMNS:
        df[col] = df[col].apply(parse_list_column)

    df["Applicants"] = df["Applicants"].apply(
        lambda lst: [normalize_applicant_name(a) for a in lst]
    )
    all_applicants_raw = sorted(set(a for row in df["Applicants"] for a in row if a))
    applicant_name_map, merge_pairs = fuzzy_merge_applicants(all_applicants_raw, threshold=fuzzy_threshold)
    df["Applicants"] = df["Applicants"].apply(
        lambda lst: [applicant_name_map.get(a, a) for a in lst]
    )

    for col in LIST_COLUMNS:
        df[col] = df[col].apply(ensure_list)

    applicant_to_id = make_mapping(a for row in df["Applicants"] for a in row)
    inventor_to_id = make_mapping(i for row in df["Inventors"] for i in row)
    cpc_to_id = make_mapping(c for row in df["CPC Classifications"] for c in row)
    patent_to_id = make_mapping(
        p for _, row in df.iterrows() for p in (row["Priority Numbers"] or [row["Lens ID"]])
    )

    merge_log_df = pd.DataFrame(merge_pairs, columns=["variant", "canonical", "score"])
    nodes_df = pd.DataFrame({"node_id": applicant_to_id.values(), "name": applicant_to_id.keys()})
    years = sorted(df["Publication Year"].dropna().unique())

    data_by_year = {
        year: build_pyg_hetero_graph(
            df[df["Publication Year"] == year],
            applicant_to_id, inventor_to_id, patent_to_id, cpc_to_id
        )
        for year in years
    }
    data_by_year_cumulative = {
        year: build_pyg_hetero_graph(
            df[df["Publication Year"] <= year],
            applicant_to_id, inventor_to_id, patent_to_id, cpc_to_id
        )
        for year in years
    }

    return {
        "raw_data": raw_data,
        "df": df,
        "merge_log_df": merge_log_df,
        "nodes_df": nodes_df,
        "applicant_to_id": applicant_to_id,
        "inventor_to_id": inventor_to_id,
        "cpc_to_id": cpc_to_id,
        "patent_to_id": patent_to_id,
        "years": years,
        "data_by_year": data_by_year,
        "data_by_year_cumulative": data_by_year_cumulative,
    }
