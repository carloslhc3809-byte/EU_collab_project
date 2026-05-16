from prefect import flow, task
from prefect.logging import get_run_logger

from src.config import DataConfig, FeatureConfig, ModelConfig
from src.data import load_and_prepare_dataset, resolve_data_path
from src.export import save_pipeline_outputs
from src.features import build_temporal_tensor, generate_temporal_embeddings
from src.train import train_temporal_link_predictor
from src.utils import get_device, set_seed


@task
def initialize_pipeline():
    logger = get_run_logger()

    logger.info("Loading configuration...")
    data_cfg = DataConfig()
    feature_cfg = FeatureConfig()
    model_cfg = ModelConfig()

    logger.info("Creating output directories...")
    data_cfg.ensure_dirs()

    logger.info(f"Setting random seed to {model_cfg.random_seed}...")
    set_seed(model_cfg.random_seed)

    logger.info("Detecting device...")
    device = get_device()
    logger.info(f"Using device: {device}")

    return data_cfg, feature_cfg, model_cfg, device


@task(retries=1, retry_delay_seconds=10)
def load_dataset(data_cfg: DataConfig):
    logger = get_run_logger()

    logger.info("Resolving raw data path...")
    data_path = resolve_data_path(data_cfg.raw_candidates)
    logger.info(f"Dataset found at: {data_path}")

    logger.info("Loading and preparing dataset...")
    bundle = load_and_prepare_dataset(data_path)
    logger.info("Dataset preparation complete.")

    logger.info("Saving merge log and node metadata...")
    bundle["merge_log_df"].to_csv(
        data_cfg.processed_dir / "applicant_name_merges.csv",
        index=False,
    )
    bundle["nodes_df"].to_csv(
        data_cfg.metadata_dir / "nodes_master.csv",
        index=False,
    )

    return data_path, bundle


@task
def create_temporal_embeddings(bundle: dict, feature_cfg: FeatureConfig, device):
    logger = get_run_logger()

    logger.info("Generating temporal embeddings...")
    emb_by_year = generate_temporal_embeddings(
        data_by_year=bundle["data_by_year"],
        device=device,
        cfg=feature_cfg,
    )
    logger.info("Temporal embeddings generated.")

    return emb_by_year


@task
def create_temporal_tensor(emb_by_year: dict, bundle: dict):
    logger = get_run_logger()

    logger.info("Building temporal tensor...")
    X, years_sorted = build_temporal_tensor(
        emb_by_year,
        bundle["applicant_to_id"],
    )

    logger.info(
        f"Temporal tensor built for {len(years_sorted)} years: "
        f"{years_sorted[0]}-{years_sorted[-1]}"
    )

    return X, years_sorted


@task
def train_model(X, bundle: dict, model_cfg: ModelConfig, device):
    logger = get_run_logger()

    logger.info("Training temporal link predictor...")
    artifacts = train_temporal_link_predictor(
        X=X,
        data_by_year=bundle["data_by_year"],
        applicant_to_id=bundle["applicant_to_id"],
        device=device,
        cfg=model_cfg,
    )
    logger.info("Model training complete.")

    return artifacts


@task
def export_pipeline_outputs(
    artifacts: dict,
    bundle: dict,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
):
    logger = get_run_logger()

    logger.info("Saving pipeline outputs...")
    save_pipeline_outputs(
        artifacts=artifacts,
        applicant_to_id=bundle["applicant_to_id"],
        cfg=data_cfg,
        model_cfg=model_cfg,
    )
    logger.info("Outputs saved.")

    return {
        "final_auc": artifacts["metrics"]["final_auc"],
        "final_ap": artifacts["metrics"]["final_ap"],
    }


@flow(name="EU Life Sciences Co-Patent Temporal GNN Pipeline")
def main() -> None:
    logger = get_run_logger()

    logger.info("Starting pipeline...")

    data_cfg, feature_cfg, model_cfg, device = initialize_pipeline()

    data_path, bundle = load_dataset(data_cfg)

    emb_by_year = create_temporal_embeddings(
        bundle=bundle,
        feature_cfg=feature_cfg,
        device=device,
    )

    X, years_sorted = create_temporal_tensor(
        emb_by_year=emb_by_year,
        bundle=bundle,
    )

    artifacts = train_model(
        X=X,
        bundle=bundle,
        model_cfg=model_cfg,
        device=device,
    )

    metrics = export_pipeline_outputs(
        artifacts=artifacts,
        bundle=bundle,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
    )

    logger.info("Pipeline complete.")
    logger.info(f"Dataset: {data_path}")
    logger.info(f"Years used: {years_sorted[0]}-{years_sorted[-1]}")
    logger.info(f"Final AUC: {metrics['final_auc']:.4f}")
    logger.info(f"Final AP: {metrics['final_ap']:.4f}")


if __name__ == "__main__":
    main()
