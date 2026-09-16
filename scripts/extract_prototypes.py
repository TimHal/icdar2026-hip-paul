#!/usr/bin/env python3
"""Extract prototype images from torchvision datasets.

Extracts N random samples per class from a dataset and saves them as images
in a folder structure compatible with PrototypeProvider.from_folder().

Usage:
    python scripts/extract_prototypes.py --dataset MNIST --output_dir ./data/prototypes/mnist_random_3 --n_prototypes 3
    python scripts/extract_prototypes.py --dataset EMNIST --emnist_split balanced --output_dir ./data/prototypes/emnist_balanced_5 --n_prototypes 5 --seed 123
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
from torchvision import datasets

SUPPORTED_DATASETS = [
    "MNIST",
    "FashionMNIST",
    "KMNIST",
    "EMNIST",
    "QMNIST",
    "CIFAR10",
    "CIFAR100",
]

EMNIST_SPLITS = ["balanced", "byclass", "bymerge", "digits", "letters", "mnist"]


def get_dataset(name: str, root: str, emnist_split: str = "digits"):
    """Load a torchvision dataset without transforms (raw PIL images).

    Args:
        name: Dataset name (e.g., 'MNIST', 'EMNIST')
        root: Root directory for dataset storage
        emnist_split: EMNIST split if using EMNIST

    Returns:
        Torchvision dataset returning (PIL.Image, label)
    """
    kwargs = {"root": root, "train": True, "transform": None, "download": True}

    if name == "MNIST":
        return datasets.MNIST(**kwargs)
    elif name == "FashionMNIST":
        return datasets.FashionMNIST(**kwargs)
    elif name == "KMNIST":
        return datasets.KMNIST(**kwargs)
    elif name == "EMNIST":
        return datasets.EMNIST(**kwargs, split=emnist_split)
    elif name == "QMNIST":
        return datasets.QMNIST(root=root, what="train", transform=None, download=True)
    elif name == "CIFAR10":
        return datasets.CIFAR10(**kwargs)
    elif name == "CIFAR100":
        return datasets.CIFAR100(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}. Supported: {SUPPORTED_DATASETS}")


def get_num_classes(dataset_name: str, emnist_split: str = "digits") -> int:
    """Get number of classes for a dataset.

    Args:
        dataset_name: Name of the dataset
        emnist_split: EMNIST split if using EMNIST

    Returns:
        Number of classes
    """
    class_counts = {
        "MNIST": 10,
        "FashionMNIST": 10,
        "KMNIST": 10,
        "QMNIST": 10,
        "CIFAR10": 10,
        "CIFAR100": 100,
    }

    if dataset_name == "EMNIST":
        emnist_class_counts = {
            "balanced": 47,
            "byclass": 62,
            "bymerge": 47,
            "digits": 10,
            "letters": 26,
            "mnist": 10,
        }
        return emnist_class_counts.get(emnist_split, 47)

    return class_counts.get(dataset_name, 10)


def select_prototypes(
    dataset, num_classes: int, n_prototypes: int, seed: int
) -> dict[int, list[int]]:
    """Randomly select prototype indices for each class.

    Args:
        dataset: Dataset to select from
        num_classes: Number of classes in dataset
        n_prototypes: Number of prototypes per class
        seed: Random seed for reproducibility

    Returns:
        Dict mapping class_idx -> list of sample indices
    """
    random.seed(seed)

    indices_by_class = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if isinstance(label, int):
            indices_by_class[label].append(idx)
        else:
            indices_by_class[int(label)].append(idx)

    selected = {}
    for class_idx in range(num_classes):
        class_indices = indices_by_class.get(class_idx, [])
        if len(class_indices) == 0:
            print(f"Warning: No samples found for class {class_idx}")
            selected[class_idx] = []
        elif len(class_indices) < n_prototypes:
            print(
                f"Warning: Class {class_idx} has only {len(class_indices)} samples, "
                f"using all of them"
            )
            selected[class_idx] = class_indices
        else:
            selected[class_idx] = random.sample(class_indices, n_prototypes)

    return selected


def save_prototypes(
    dataset,
    indices_by_class: dict[int, list[int]],
    output_dir: Path,
    img_size: int | None,
    img_mode: str,
    dataset_name: str,
):
    """Save selected prototypes as images in folder structure.

    Args:
        dataset: Source dataset
        indices_by_class: Dict mapping class_idx -> list of sample indices
        output_dir: Output directory
        img_size: Optional resize dimension (None = original size)
        img_mode: Image mode ('L' for grayscale, 'RGB' for color)
        dataset_name: Name of dataset (for EMNIST rotation handling)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total_saved = 0

    for class_idx, indices in sorted(indices_by_class.items()):
        class_dir = output_dir / str(class_idx)
        class_dir.mkdir(exist_ok=True)

        for proto_idx, sample_idx in enumerate(indices):
            img, _ = dataset[sample_idx]

            if not isinstance(img, Image.Image):
                raise ValueError(f"Expected PIL Image, got {type(img)}")

            # EMNIST images need rotation/flip correction
            if dataset_name == "EMNIST":
                img = img.transpose(Image.TRANSPOSE)

            # Convert to target mode
            if img.mode != img_mode:
                img = img.convert(img_mode)

            # Resize if specified
            if img_size is not None:
                img = img.resize((img_size, img_size), Image.BILINEAR)

            # Save
            output_path = class_dir / f"proto_{proto_idx}.png"
            img.save(output_path)
            total_saved += 1

    return total_saved


def main():
    parser = argparse.ArgumentParser(
        description="Extract prototype images from torchvision datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Dataset to extract prototypes from",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for prototype images",
    )
    parser.add_argument(
        "--n_prototypes",
        type=int,
        default=1,
        help="Number of prototypes per class (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=None,
        help="Resize dimension (default: original size)",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/data/thallybu/datasets",
        help="Root directory for datasets",
    )
    parser.add_argument(
        "--img_mode",
        type=str,
        default="L",
        choices=["L", "RGB"],
        help="Image mode: L=grayscale, RGB=color (default: L)",
    )
    parser.add_argument(
        "--emnist_split",
        type=str,
        default="digits",
        choices=EMNIST_SPLITS,
        help="EMNIST split to use (default: digits)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print(f"Dataset: {args.dataset}")
    if args.dataset == "EMNIST":
        print(f"EMNIST split: {args.emnist_split}")
    print(f"Output: {output_dir}")
    print(f"Prototypes per class: {args.n_prototypes}")
    print(f"Seed: {args.seed}")
    print(f"Image mode: {args.img_mode}")
    if args.img_size:
        print(f"Resize to: {args.img_size}x{args.img_size}")

    print("\nLoading dataset...")
    dataset = get_dataset(args.dataset, args.root_dir, args.emnist_split)
    num_classes = get_num_classes(args.dataset, args.emnist_split)
    print(f"Loaded {len(dataset)} samples, {num_classes} classes")

    print("\nSelecting prototypes...")
    indices = select_prototypes(dataset, num_classes, args.n_prototypes, args.seed)

    print("Saving prototypes...")
    total = save_prototypes(
        dataset, indices, output_dir, args.img_size, args.img_mode, args.dataset
    )

    print(f"\nDone! Saved {total} prototype images to {output_dir}")
    print(f"Structure: {output_dir}/{{class_idx}}/proto_{{i}}.png")


if __name__ == "__main__":
    main()
