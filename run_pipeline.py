from src.config import DataConfig, FeatureConfig, ModelConfig
from src.data import load_and_prepare_dataset, resolve_data_path
from src.export import save_pipeline_outputs
from src.features import build_temporal_tensor, generate_temporal_embeddings
from src.train import train_temporal_link_predictor
from src.utils import get_device, set_seed


def main() -> None:
    print("Starting pipeline...")

    print("Loading configuration...")
    data_cfg = DataConfig()
    feature_cfg = FeatureConfig()
    model_cfg = ModelConfig()

    print("Creating output directories...")
    data_cfg.ensure_dirs()

    print(f"Setting random seed to {model_cfg.random_seed}...")
    set_seed(model_cfg.random_seed)

    print("Detecting device...")
    device = get_device()
    print(f"Using device: {device}")

    print("Resolving raw data path...")
    data_path = resolve_data_path(data_cfg.raw_candidates)
    print(f"Dataset found at: {data_path}")

    print("Loading and preparing dataset...")
    bundle = load_and_prepare_dataset(data_path)
    print("Dataset preparation complete.")

    print("Saving merge log and node metadata...")
    bundle["merge_log_df"].to_csv(data_cfg.processed_dir / "applicant_name_merges.csv", index=False)
    bundle["nodes_df"].to_csv(data_cfg.metadata_dir / "nodes_master.csv", index=False)

    print("Generating temporal embeddings...")
    emb_by_year = generate_temporal_embeddings(
        data_by_year=bundle["data_by_year"],
        device=device,
        cfg=feature_cfg,
    )
    print("Temporal embeddings generated.")

    print("Building temporal tensor...")
    X, years_sorted = build_temporal_tensor(emb_by_year, bundle["applicant_to_id"])
    print(f"Temporal tensor built for {len(years_sorted)} years: {years_sorted[0]}-{years_sorted[-1]}")

    print("Training temporal link predictor...")
    artifacts = train_temporal_link_predictor(
        X=X,
        data_by_year=bundle["data_by_year"],
        applicant_to_id=bundle["applicant_to_id"],
        device=device,
        cfg=model_cfg,
    )
    print("Model training complete.")

    print("Saving pipeline outputs...")
    save_pipeline_outputs(
        artifacts=artifacts,
        applicant_to_id=bundle["applicant_to_id"],
        cfg=data_cfg,
        model_cfg=model_cfg,
    )
    print("Outputs saved.")

    print("Pipeline complete.")
    print(f"Dataset: {data_path}")
    print(f"Years used: {years_sorted[0]}-{years_sorted[-1]}")
    print(f"Final AUC: {artifacts['metrics']['final_auc']:.4f}")
    print(f"Final AP: {artifacts['metrics']['final_ap']:.4f}")


if __name__ == "__main__":
    main()
