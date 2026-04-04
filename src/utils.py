import random
from typing import Any

import numpy as np
import torch


LIST_COLUMNS = ["Applicants", "Inventors", "CPC Classifications", "Priority Numbers"]

TOKEN_REPLACEMENTS = {
    "biotechnologies": "biotechnology",
    "technologies": "technology",
    "therapeutics": "therapeutic",
    "laboratories": "laboratory",
    "systems": "system",
    "sciences": "science",
    "holdings": "holding",
    "communications": "communication",
    "solutions": "solution",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_python(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, list):
        return [to_python(v) for v in value]
    if isinstance(value, tuple):
        return [to_python(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_python(v) for k, v in value.items()}
    return value
