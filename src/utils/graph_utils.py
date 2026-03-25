import torch


def build_global_app_map(app_maps):
    all_apps = set()

    for app_map in app_maps.values():
        all_apps.update(app_map.keys())

    global_apps = sorted(all_apps)
    global_map = {app: i for i, app in enumerate(global_apps)}

    return global_apps, global_map


def align_embeddings_to_global(emb_by_year, app_maps, global_map):

    aligned = {}

    for year, emb in emb_by_year.items():

        app_map = app_maps[year]
        num_nodes = len(global_map)
        dim = emb.shape[1]

        Z = torch.zeros((num_nodes, dim))

        for app, local_idx in app_map.items():

            if app not in global_map:
                continue

            global_idx = global_map[app]
            Z[global_idx] = emb[local_idx]

        aligned[year] = Z

    return aligned


def build_app_maps_from_embeddings(emb_by_year):

    app_maps = {}

    for year, emb in emb_by_year.items():
        num_nodes = emb.shape[0]
        app_maps[year] = {i: i for i in range(num_nodes)}

    return app_maps


def stack_temporal_embeddings(aligned_emb_by_year):

    years = sorted(aligned_emb_by_year.keys())

    T = len(years)
    N = list(aligned_emb_by_year.values())[0].shape[0]
    F = list(aligned_emb_by_year.values())[0].shape[1]

    X = torch.zeros((T, N, F))

    for t, year in enumerate(years):
        X[t] = aligned_emb_by_year[year]

    return X, years


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj
