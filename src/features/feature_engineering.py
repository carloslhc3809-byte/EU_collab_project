import torch
from torch_geometric.nn import MetaPath2Vec


def get_metapaths(data, metapath_type="ALL"):

    APA = [
        ("applicant", "files", "patent"),
        ("patent", "rev_files", "applicant")
    ]

    API = [
        ("applicant", "files", "patent"),
        ("patent", "rev_invented", "inventor"),
        ("inventor", "invented", "patent"),
        ("patent", "rev_files", "applicant")
    ]

    APC = [
        ("applicant", "files", "patent"),
        ("patent", "rev_in_patent", "cpc"),
        ("cpc", "in_patent", "patent"),
        ("patent", "rev_files", "applicant")
    ]

    if metapath_type == "ALL":
        return [APA, API, APC]
    elif metapath_type == "NO_API":
        return [APA, APC]
    elif metapath_type == "NO_APC":
        return [APA, API]
    else:
        return [APA]


def generate_metapath2vec_embeddings(
    data,
    metapath_type="ALL",
    embedding_dim=64,
    walk_length=20,
    context_size=10,
    walks_per_node=5,
    num_negative_samples=5,
    epochs=5,
    device="cpu"
):

    metapaths = get_metapaths(data, metapath_type)

    model = MetaPath2Vec(
        data.edge_index_dict,
        embedding_dim=embedding_dim,
        metapath=metapaths,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=num_negative_samples,
        sparse=True
    ).to(device)

    loader = model.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    model.train()
    for _ in range(epochs):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return model


def extract_applicant_embeddings(model, data):
    return model("applicant").detach().cpu()


def generate_embeddings(data_by_year, metapath_type="ALL", embedding_dim=64):

    emb_by_year = {}

    for year, data in data_by_year.items():

        model = generate_metapath2vec_embeddings(
            data,
            metapath_type=metapath_type,
            embedding_dim=embedding_dim
        )

        emb = extract_applicant_embeddings(model, data)
        emb_by_year[year] = emb

    return emb_by_year


def build_temporal_tensor(emb_by_year):

    years = sorted(emb_by_year.keys())
    T = len(years)
    num_nodes = max(emb.shape[0] for emb in emb_by_year.values())
    dim = list(emb_by_year.values())[0].shape[1]

    X = torch.zeros((T, num_nodes, dim))

    for t, year in enumerate(years):
        emb = emb_by_year[year]
        X[t, :emb.shape[0], :] = emb

    return X, years
