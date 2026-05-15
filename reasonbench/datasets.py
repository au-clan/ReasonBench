"""Utility for downloading datasets from HuggingFace Hub."""

import os
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

HF_REPO_ID = "potamitisn/ReasonBench"
LOCAL_DATASETS_DIR = Path("datasets")

# Maps task name to the actual filename on HF Hub
DATASET_FILES = {
    "game24": "dataset_game24.csv.gz",
    "hle": "dataset_hle.jsonl.gz",
    "hotpotqa": "dataset_hotpotqa.csv.gz",
    "humaneval": "dataset_humaneval.csv.gz",
    "logiqa": "dataset_logiqa.csv.gz",
    "matharena": "dataset_matharena.jsonl.gz",
    "scibench": "dataset_scibench.csv.gz",
    "sonnetwriting": "dataset_sonnetwriting.jsonl.gz",
}


def get_dataset_path(task: str, local_dir: str | Path = LOCAL_DATASETS_DIR) -> str:
    """Return the local path to a dataset file, downloading from HF Hub if needed.

    Args:
        task: Task name (e.g., "game24", "hle").
        local_dir: Local directory to store/look for datasets.

    Returns:
        Path to the local dataset file.
    """
    if task not in DATASET_FILES:
        raise ValueError(f"Unknown task: {task}. Available: {list(DATASET_FILES.keys())}")

    filename = DATASET_FILES[task]
    local_path = Path(local_dir) / filename

    if local_path.exists():
        return str(local_path)

    logger.info(f"Dataset for '{task}' not found locally. Downloading from HuggingFace Hub...")
    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    logger.info(f"Downloaded to {downloaded}")
    return str(downloaded)
