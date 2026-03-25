import pandas as pd
import ast
import torch
from torch_geometric.data import HeteroData


def parse_list_column(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except:
        return [i.strip() for i in str(x).split(";")]


def load_raw_data(path):
    df = pd.read_csv(path)
    df["family_id"] = df["Simple Family Members"].str.split(";;").str[0]
    df = df.drop_duplicates(subset="family_id")
    return df


def preprocess_dataframe(df, max_year=2025):
    df = df[df["Publication Year"] < max_year].copy()

    list_cols = [
        "Applicants",
        "Inventors",
        "CPC Classifications",
        "Priority Numbers"
    ]

    for col in list_cols:
        df[col] = df[col].apply(parse_list_column)

    return df


def build_pyg_hetero_graph(df):
    data = HeteroData()

    applicants = set()
    inventors = set()
    patents = set()
    cpcs = set()

    for _, row in df.iterrows():
        pats = row["Priority Numbers"] or [row["Lens ID"]]

        for p in pats:
            patents.add(p)

        applicants.update(row["Applicants"])
        inventors.update(row["Inventors"])
        cpcs.update(row["CPC Classifications"])

    app_map = {a: i for i, a in enumerate(applicants)}
    inv_map = {i: j for j, i in enumerate(inventors)}
    pat_map = {p: i for i, p in enumerate(patents)}
    cpc_map = {c: i for i, c in enumerate(cpcs)}

    data["applicant"].num_nodes = len(app_map)
    data["inventor"].num_nodes = len(inv_map)
    data["patent"].num_nodes = len(pat_map)
    data["cpc"].num_nodes = len(cpc_map)

    edge_dict = {
        ("applicant", "files", "patent"): [],
        ("inventor", "invented", "patent"): [],
        ("cpc", "in_patent", "patent"): [],
    }

    for _, row in df.iterrows():
        pats = row["Priority Numbers"] or [row["Lens ID"]]

        for p in pats:
            p_idx = pat_map[p]

            for a in row["Applicants"]:
                if a in app_map:
                    edge_dict[("applicant", "files", "patent")].append(
                        (app_map[a], p_idx)
                    )

            for i in row["Inventors"]:
                if i in inv_map:
                    edge_dict[("inventor", "invented", "patent")].append(
                        (inv_map[i], p_idx)
                    )

            for c in row["CPC Classifications"]:
                if c in cpc_map:
                    edge_dict[("cpc", "in_patent", "patent")].append(
                        (cpc_map[c], p_idx)
                    )

    for (src, rel, dst), edges in edge_dict.items():
        if len(edges) == 0:
            continue

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        data[(src, rel, dst)].edge_index = edge_index

        rev_rel = f"rev_{rel}"
        data[(dst, rev_rel, src)].edge_index = edge_index.flip(0)

    return data, app_map


def build_temporal_graphs(df):
    data_by_year = {}
    app_maps = {}

    years = sorted(df["Publication Year"].dropna().unique())

    for year in years:
        df_year = df[df["Publication Year"] == year]
        data_year, app_map = build_pyg_hetero_graph(df_year)

        data_by_year[year] = data_year
        app_maps[year] = app_map

    return data_by_year, app_maps
