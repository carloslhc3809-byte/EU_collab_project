from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class DataConfig:
    repo_root: Path = Path(__file__).resolve().parents[1]
    raw_candidates: List[Path] = field(default_factory=list)
    processed_dir: Path = Path()
    metadata_dir: Path = Path()
    embeddings_dir: Path = Path()
    predictions_dir: Path = Path()
    edges_dir: Path = Path()

    def __post_init__(self) -> None:
        if not self.raw_candidates:
            self.raw_candidates = [
                self.repo_root / "data" / "raw" / "EU_C12N15_data.csv",
                self.repo_root / "data" / "sample.csv",
            ]
        self.processed_dir = self.repo_root / "data" / "processed"
        self.metadata_dir = self.processed_dir / "metadata"
        self.embeddings_dir = self.processed_dir / "embeddings"
        self.predictions_dir = self.processed_dir / "predictions"
        self.edges_dir = self.processed_dir / "edges"

    def ensure_dirs(self) -> None:
        for path in [
            self.processed_dir,
            self.metadata_dir,
            self.embeddings_dir,
            self.predictions_dir,
            self.edges_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class FeatureConfig:
    emb_dim: int = 32
    metapaths: Dict[str, Dict] = field(default_factory=lambda: {
        "APA": {
            "path": [
                ("applicant", "files", "patent"),
                ("patent", "rev_files", "applicant"),
            ],
            "weight": 1.0,
        },
        "API": {
            "path": [
                ("applicant", "files", "patent"),
                ("patent", "rev_invented", "inventor"),
                ("inventor", "invented", "patent"),
                ("patent", "rev_files", "applicant"),
            ],
            "weight": 1.2,
        },
        "APC": {
            "path": [
                ("applicant", "files", "patent"),
                ("patent", "rev_in_patent", "cpc"),
                ("cpc", "in_patent", "patent"),
                ("patent", "rev_files", "applicant"),
            ],
            "weight": 1.2,
        },
    })
    walk_length: int = 30
    context_size: int = 5
    walks_per_node: int = 3
    num_negative_samples: int = 5
    batch_size: int = 128
    epochs: int = 2


@dataclass
class ModelConfig:
    gnn_hidden_dim: int = 128
    rnn_hidden_dim: int = 128
    gnn_layers: int = 2
    dropout: float = 0.2
    use_attention: bool = True
    predictor: str = "mlp"
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 300
    grad_clip: float = 1.0
    eval_every: int = 5
    emb_penalty_weight: float = 1e-3
    sigmoid_clip: float = 8.0
    random_seed: int = 42
    k_clusters: int = 10
    top_k_predictions: int = 50
