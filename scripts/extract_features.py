#!/usr/bin/env python3
"""Extract and cache features from foundation models for autoencoder training.

Usage examples:
    python scripts/extract_features.py --dataset MNIST --model dinov2_vits14 --output ./cache
    python scripts/extract_features.py --data_dir /data/thallybu/datasets/custom/26ICDAR-RERC/OMNIGLOT20 \
        --model dinov2_vits14 --output ./cache
"""

import argparse
import sys
from pathlib import Path

from feature_extractors.mae import MAEFeatureExtractor
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

# Ensure src is on path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feature_extractors import DINOFeatureExtractor, CLIPFeatureExtractor

SUPPORTED_DATASETS = [
    "MNIST",
    "FashionMNIST",
    "KMNIST",
    "EMNIST",
    "QMNIST",
    "CIFAR10",
    "CIFAR100",
    "STL10",
]

EMNIST_SPLITS = ["balanced", "byclass", "bymerge", "digits", "letters", "mnist"]


def get_dataset(name: str, root: str, train: bool, transform, emnist_split: str = "digits"):
    """Load a torchvision dataset."""
    kwargs = {"root": root, "train": train, "transform": transform, "download": True}
    if name == "MNIST":
        return datasets.MNIST(**kwargs)
    if name == "FashionMNIST":
        return datasets.FashionMNIST(**kwargs)
    if name == "KMNIST":
        return datasets.KMNIST(**kwargs)
    if name == "EMNIST":
        return datasets.EMNIST(**kwargs, split=emnist_split)
    if name == "QMNIST":
        split = "train" if train else "test"
        return datasets.QMNIST(root=root, what=split, transform=transform, download=True)
    if name == "CIFAR10":
        return datasets.CIFAR10(**kwargs)
    if name == "CIFAR100":
        return datasets.CIFAR100(**kwargs)
    if name == "STL10":
        split = "train" if train else "test"
        return datasets.STL10(root=root, split=split, transform=transform, download=True)
    raise ValueError(f"Unknown dataset: {name}. Supported: {SUPPORTED_DATASETS}")


def get_imagefolder_split(root: Path, split: str, transform):
    """Load a split from an ImageFolder-style dataset that already has train/test subfolders."""
    split_dir = root / split
    if not split_dir.is_dir():
        raise ValueError(f"Expected subdirectory '{split}' under {root}")
    return datasets.ImageFolder(str(split_dir), transform=transform)


def get_extractor(
    model_name: str, pretrained: str = "openai", device: torch.device = None, ckpt_path: str = None
):
    """Instantiate the appropriate feature extractor."""
    if model_name.startswith("dino"):
        return DINOFeatureExtractor(model_name=model_name, device=device)
    if model_name == "mae":
        if ckpt_path is None:
            raise ValueError("ckpt_path must be provided for mae feature extractor")
        return MAEFeatureExtractor.from_checkpoint(ckpt_path, device=device)
    # CLIP case
    if ckpt_path is not None:
        return CLIPFeatureExtractor.from_checkpoint(model_name, ckpt_path, device=device)
    return CLIPFeatureExtractor(model_name=model_name, pretrained=pretrained, device=device)


def get_transform(extractor, dataset_name: str, image_size: int = 224):
    """Return appropriate transform, handling grayscale datasets."""
    grayscale = ["MNIST", "FashionMNIST", "KMNIST", "EMNIST", "QMNIST", "Custom"]
    is_gray = dataset_name in grayscale
    if isinstance(extractor, DINOFeatureExtractor):
        return (
            extractor.get_transform_for_grayscale(image_size)
            if is_gray
            else extractor.get_transform(image_size)
        )
    if isinstance(extractor, CLIPFeatureExtractor):
        return (
            extractor.get_transform_for_grayscale(image_size)
            if is_gray
            else extractor.get_transform()
        )
    if isinstance(extractor, MAEFeatureExtractor):
        return (
            extractor.get_transform_for_grayscale(image_size)
            if is_gray
            else extractor.get_transform(image_size)
        )
    raise ValueError(f"Unknown extractor type: {type(extractor)}")


def main():
    parser = argparse.ArgumentParser(description="Extract features from foundation models")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=SUPPORTED_DATASETS,
        help="Standard torchvision dataset (use --data_dir for custom)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to custom ImageFolder dataset with train/ and test/ subfolders",
    )
    parser.add_argument(
        "--model", type=str, default="dinov2_vits14", help="Feature extractor model name"
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Checkpoint path for mae extractor"
    )
    parser.add_argument(
        "--pretrained", type=str, default="openai", help="Pretrained weights for CLIP models"
    )
    parser.add_argument(
        "--output", type=str, default="./cache", help="Output directory for .pt files"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/thallybu/datasets",
        help="Root for torchvision datasets",
    )
    parser.add_argument(
        "--emnist_split", type=str, default="digits", choices=EMNIST_SPLITS, help="EMNIST split"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--image_size", type=int, default=224, help="Image size for extractor")
    parser.add_argument(
        "--normalize", action="store_true", default=True, help="Apply MinMax normalization"
    )
    args = parser.parse_args()

    if (args.dataset is None) == (args.data_dir is None):
        raise ValueError("Specify exactly one of --dataset or --data_dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = get_extractor(args.model, args.pretrained, device, ckpt_path=args.ckpt_path)
    print(f"Loaded model: {args.model}, feature dim: {extractor.feature_dim}")

    # Determine transform dataset name for grayscale handling
    transform_name = args.dataset if args.dataset is not None else "Custom"
    transform = get_transform(extractor, transform_name, args.image_size)

    # Build output filename prefix
    model_clean = extractor.model_name.replace("/", "-").replace(":", "-")
    if args.data_dir:
        dataset_name = Path(args.data_dir).name
    else:
        dataset_name = (
            f"{args.dataset}_{args.emnist_split}" if args.dataset == "EMNIST" else args.dataset
        )
    prefix = f"{dataset_name.lower()}_{model_clean.lower()}"

    for split_name, is_train in [("train", True), ("test", False)]:
        print(f"\nExtracting {split_name} split...")
        if args.data_dir:
            root = Path(args.data_dir)
            dataset = get_imagefolder_split(root, split_name, transform)
        else:
            dataset = get_dataset(
                args.dataset,
                args.data_root,
                train=is_train,
                transform=transform,
                emnist_split=args.emnist_split,
            )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        features, labels = extractor.extract_from_dataloader(loader, device)
        print(f"Features shape: {features.shape}")
        if args.normalize:
            if split_name == "train":
                features, norm_params = extractor.normalize_features(features, method="minmax")
                train_norm_params = norm_params
            else:
                features = extractor.apply_normalization(features, train_norm_params)
            print("Normalization applied")
        num_classes = len(torch.unique(labels))
        out_file = out_dir / f"{prefix}_{split_name}.pt"
        torch.save(
            {
                "features": features,
                "labels": labels,
                "metadata": {
                    "dataset": args.dataset,
                    "emnist_split": args.emnist_split if args.dataset == "EMNIST" else None,
                    "model": args.model,
                    "pretrained": args.pretrained if not args.model.startswith("dino") else None,
                    "feature_dim": extractor.feature_dim,
                    "num_samples": len(features),
                    "num_classes": num_classes,
                    "normalized": args.normalize,
                    "split": split_name,
                },
                **(
                    {"norm_params": train_norm_params}
                    if args.normalize and split_name == "train"
                    else {}
                ),
            },
            out_file,
        )
        print(f"Saved to {out_file}")

    print("\nFeature extraction complete!")
    print(f"Files stored in {out_dir}")


if __name__ == "__main__":
    main()
