import torch

from src.data.preprocessing import (
    load_raw_data,
    preprocess_dataframe,
    build_temporal_graphs
)

from src.features.feature_engineering import (
    generate_embeddings
)

from src.utils.graph_utils import (
    build_global_app_map,
    align_embeddings_to_global,
    stack_temporal_embeddings
)

from src.training.train_link_prediction import (
    set_seed,
    normalize_features,
    build_train_test_edges,
    train,
    evaluate
)

from src.models.models import TemporalLinkPredictor


DATA_PATH = "data/raw/EU_C12N15_data.csv"


def run_data():
    print("\n📦 Step 1: Data Preparation\n")

    df = load_raw_data(DATA_PATH)
    df = preprocess_dataframe(df)

    data_by_year, app_maps = build_temporal_graphs(df)

    torch.save(data_by_year, "data/processed/data_by_year.pt")
    torch.save(app_maps, "data/processed/app_maps.pt")

    print("✅ Data saved")


def run_features():
    print("\n🧠 Step 2: Feature Engineering\n")

    data_by_year = torch.load("data/processed/data_by_year.pt")

    emb_by_year = generate_embeddings(
        data_by_year,
        metapath_type="ALL",
        embedding_dim=64
    )

    torch.save(emb_by_year, "data/processed/emb_by_year.pt")

    print("✅ Embeddings saved")


def run_training():
    print("\n🚀 Step 3: Training & Evaluation\n")

    data_by_year = torch.load("data/processed/data_by_year.pt")
    app_maps = torch.load("data/processed/app_maps.pt")
    emb_by_year = torch.load("data/processed/emb_by_year.pt")

    global_apps, global_map = build_global_app_map(app_maps)

    aligned = align_embeddings_to_global(
        emb_by_year,
        app_maps,
        global_map
    )

    X, years = stack_temporal_embeddings(aligned)

    train_edge_index, test_edge_index, train_T = build_train_test_edges(
        data_by_year,
        app_maps,
        global_apps
    )

    X, X_train = normalize_features(X, train_T)

    model = TemporalLinkPredictor(
        in_dim=X.shape[-1],
        hid=64,
        predictor="mlp",
        temporal="attention"
    )

    model = train(model, X, train_edge_index)

    auc, ap = evaluate(model, X, test_edge_index)

    print("\n📊 FINAL PERFORMANCE")
    print(f"AUC: {auc:.4f}")
    print(f"AP:  {ap:.4f}")


def main():

    set_seed(42)

    run_data()
    run_features()
    run_training()


if __name__ == "__main__":
    main()
