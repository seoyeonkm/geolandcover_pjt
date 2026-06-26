"""EuroSAT preprocessing and PyTorch DataLoader utilities."""

from pathlib import Path
from typing import Dict, Tuple

from datasets import Dataset, load_dataset
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class HFDatasetWithTransforms(TorchDataset):
    """Wrap Hugging Face dataset so torchvision transforms are applied lazily."""

    def __init__(self, hf_dataset: Dataset, transform: transforms.Compose) -> None:
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Tuple:
        item = self.hf_dataset[idx]
        image = item["image"]
        label = item["label"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        image = self.transform(image)
        return image, label


def _load_eurosat_full_dataset() -> Dataset:
    """Load one usable EuroSAT split from Hugging Face.

    For beginner projects, we keep this logic simple:
    try a few known dataset IDs and return the first split that has
    both image and label columns.
    """
    candidate_ids = [
        "timm/eurosat-rgb",
        "tanganke/eurosat",
        "blanchon/EuroSAT_RGB",
    ]

    last_error = None
    for dataset_id in candidate_ids:
        try:
            ds_dict = load_dataset(dataset_id)
        except Exception as error:
            last_error = error
            continue

        # Use the first split that looks like (image, label) data.
        for split_name in ds_dict.keys():
            split_features = ds_dict[split_name].features
            if "image" in split_features and "label" in split_features:
                return ds_dict[split_name]

    raise RuntimeError(
        "Failed to load a usable EuroSAT dataset from Hugging Face. "
        "Please check internet access/HF token and dataset availability."
    ) from last_error


def _build_transforms(image_size: int) -> Dict[str, transforms.Compose]:
    """Create train/val/test transform pipelines."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return {
        "train": train_transform,
        "val": eval_transform,
        "test": eval_transform,
    }


def split_eurosat_dataset(seed: int = 42) -> Dict[str, Dataset]:
    """Split EuroSAT into train/val/test with ratio 8:1:1."""
    full_dataset = _load_eurosat_full_dataset()

    train_temp = full_dataset.train_test_split(
        test_size=0.2,
        seed=seed,
        stratify_by_column="label",
    )
    val_test = train_temp["test"].train_test_split(
        test_size=0.5,
        seed=seed,
        stratify_by_column="label",
    )

    return {
        "train": train_temp["train"],
        "val": val_test["train"],
        "test": val_test["test"],
    }


def build_eurosat_dataloaders(
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[Dict[str, DataLoader], Dict[int, str]]:
    """Build PyTorch DataLoaders for EuroSAT with augmentation.

    Returns:
        dataloaders: dict containing train/val/test DataLoader
        id_to_label: mapping from class id to class name
    """
    split_datasets = split_eurosat_dataset(seed=seed)
    tfms = _build_transforms(image_size=image_size)

    torch_datasets = {
        split_name: HFDatasetWithTransforms(hf_ds, tfms[split_name])
        for split_name, hf_ds in split_datasets.items()
    }

    use_pin_memory = torch.cuda.is_available()

    dataloaders = {
        "train": DataLoader(
            torch_datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        ),
        "val": DataLoader(
            torch_datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        ),
        "test": DataLoader(
            torch_datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        ),
    }

    features = split_datasets["train"].features["label"]
    id_to_label = {idx: name for idx, name in enumerate(features.names)}

    return dataloaders, id_to_label


def run_preprocessing(output_dir: str = "data/processed", seed: int = 42) -> None:
    """Create split metadata and verify DataLoader preparation pipeline."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    split_datasets = split_eurosat_dataset(seed=seed)
    counts_text = "\n".join(
        [f"{split_name}: {len(ds)}" for split_name, ds in split_datasets.items()]
    )

    (out_path / "split_sizes.txt").write_text(counts_text, encoding="utf-8")
    print("EuroSAT split completed (8:1:1).")
    print(counts_text)


if __name__ == "__main__":
    run_preprocessing()
