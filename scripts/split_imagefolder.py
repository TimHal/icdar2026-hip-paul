#!/usr/bin/env python3
"""Reorganize a flat ImageFolder dataset into train/test splits.

Moves images from a flat class-folder structure into train/ and test/
subdirectories as required by torchvision.datasets.ImageFolder.

Input structure (current):
    dataset_dir/class_00/0000.jpg ... 0999.jpg
    dataset_dir/class_01/0000.jpg ... 0999.jpg

Output structure (after running):
    dataset_dir/train/class_00/...
    dataset_dir/train/class_01/...
    dataset_dir/test/class_00/...
    dataset_dir/test/class_01/...

The original class directories are removed once all files have been moved.

Usage:
    python scripts/split_imagefolder.py --dataset_dir /data/.../OMNIGLOT20
    python scripts/split_imagefolder.py --dataset_dir /data/.../OMNIGLOT50 --test_ratio 0.1 --seed 0
"""

import argparse
import random
import shutil
import sys
from pathlib import Path


def collect_class_dirs(dataset_dir: Path) -> list[Path]:
    return sorted(
        p for p in dataset_dir.iterdir()
        if p.is_dir() and p.name not in {"train", "test"}
    )


def collect_images(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.iterdir() if p.is_file())


def split_indices(n: int, test_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_test = max(1, round(n * test_ratio))
    return indices[n_test:], indices[:n_test]  # (train, test)


def move_files(images: list[Path], indices: list[int], split_root: Path, class_name: str) -> int:
    dest_dir = split_root / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    for i in indices:
        shutil.move(str(images[i]), dest_dir / images[i].name)
    return len(indices)


def main():
    parser = argparse.ArgumentParser(
        description="Split a flat ImageFolder dataset into train/test subdirectories."
    )
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to the flat ImageFolder dataset root")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Fraction of images per class to use for test (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible shuffle (default: 42)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.is_dir():
        print(f"Error: {dataset_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    train_root = dataset_dir / "train"
    test_root = dataset_dir / "test"
    if train_root.exists() or test_root.exists():
        print(
            f"Warning: train/ or test/ already exists in {dataset_dir}. "
            "Skipping. Remove them manually to re-run."
        )
        sys.exit(0)

    class_dirs = collect_class_dirs(dataset_dir)
    if not class_dirs:
        print(f"Error: no class directories found in {dataset_dir}.", file=sys.stderr)
        sys.exit(1)

    records = []
    for class_dir in class_dirs:
        images = collect_images(class_dir)
        if not images:
            print(f"Warning: no files in {class_dir.name}, skipping.")
            continue

        train_idx, test_idx = split_indices(len(images), args.test_ratio, args.seed)
        n_train = move_files(images, train_idx, train_root, class_dir.name)
        n_test  = move_files(images, test_idx,  test_root,  class_dir.name)
        records.append((class_dir.name, n_train, n_test))

        # Remove now-empty original class dir
        class_dir.rmdir()

    # Print summary
    col = max(len(r[0]) for r in records)
    sep = "-" * (col + 28)
    print(f"\nSplit complete.")
    print(f"  dataset_dir : {dataset_dir}")
    print(f"  seed        : {args.seed}")
    print(f"  test_ratio  : {args.test_ratio}")
    print(f"\n  {'class':<{col}}  {'train':>8}  {'test':>6}  {'total':>7}")
    print(f"  {sep}")
    for name, n_train, n_test in records:
        print(f"  {name:<{col}}  {n_train:>8}  {n_test:>6}  {n_train + n_test:>7}")
    print(f"  {sep}")
    total_train = sum(r[1] for r in records)
    total_test  = sum(r[2] for r in records)
    print(f"  {'TOTAL':<{col}}  {total_train:>8}  {total_test:>6}  {total_train + total_test:>7}\n")


if __name__ == "__main__":
    main()
