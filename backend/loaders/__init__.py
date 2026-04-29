"""Data loaders for various robotics dataset formats."""
from .base import DatasetLoader, Episode, Modality, ModalityConfig
from .hdf5_loader import HDF5Loader
from .lerobot_loader import LeRobotLoader
from .mcap_utils import detect_mcap_modalities, list_mcap_channels
from .rlds_loader import RLDSLoader
from .webdataset_loader import WebDatasetLoader


def get_repo_id_for_dataset(dataset_id: str) -> str | None:
    """
    Get the HuggingFace repo ID for a dataset.

    Args:
        dataset_id: Internal dataset identifier

    Returns:
        HuggingFace repo ID or None if not found
    """
    # Import here to avoid circular dependency
    from downloaders.manager import get_all_datasets

    all_datasets = get_all_datasets()
    if dataset_id not in all_datasets:
        return None

    config = all_datasets[dataset_id]

    # Return explicit repo_id if set
    if "repo_id" in config:
        return config["repo_id"]

    # For LIBERO, use the LeRobot format version
    if dataset_id == "libero":
        return "HuggingFaceVLA/libero"

    return None


__all__ = [
    "DatasetLoader",
    "Episode",
    "Modality",
    "ModalityConfig",
    "HDF5Loader",
    "WebDatasetLoader",
    "LeRobotLoader",
    "RLDSLoader",
    "detect_mcap_modalities",
    "list_mcap_channels",
    "get_repo_id_for_dataset",
]
