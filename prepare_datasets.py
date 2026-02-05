# Copyright (c) Meta Platforms, Inc.
# All rights reserved.
#
# Utility script to download/convert popular datasets into the ImageFolder
# layout expected by train.py: data_path/{split}/{class_name}/image.png
#
# Supported: MNIST, CIFAR10, SVHN (train/test/+extra optional), CelebA (single class),
# ImageNet (requires manual tar download), Tiny-ImageNet (auto-downloadable fallback).

import argparse
import csv
import os
from collections import defaultdict
from typing import Iterable, List, Optional

from PIL import Image
from tqdm import tqdm

import torch
from torchvision import datasets, transforms


def _to_pil_rgb(img):
    if isinstance(img, torch.Tensor):
        img = transforms.ToPILImage()(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _save_split(dataset, split_dir: str, class_names: Optional[List[str]] = None, resize: Optional[int] = None):
    os.makedirs(split_dir, exist_ok=True)
    counts = defaultdict(int)
    resize_tf = transforms.Resize((resize, resize), interpolation=Image.BICUBIC) if resize else None
    for idx in tqdm(range(len(dataset)), desc=f"Writing {split_dir}"):
        img, label = dataset[idx]
        img = _to_pil_rgb(img)
        if resize_tf:
            img = resize_tf(img)
        cls_name = class_names[label] if class_names else str(label)
        cls_dir = os.path.join(split_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        fname = f"{counts[cls_name]:06d}.png"
        counts[cls_name] += 1
        img.save(os.path.join(cls_dir, fname))
    return counts


class SingleClassWrapper(torch.utils.data.Dataset):
    """Wraps a dataset to force a single label (e.g., CelebA for class-agnostic training)."""
    def __init__(self, base_ds, label: int = 0):
        self.base = base_ds
        self.label = label

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return img, self.label


def prepare_mnist(args, out_root: str):
    train_ds = datasets.MNIST(args.download_root, train=True, download=True)
    test_ds = datasets.MNIST(args.download_root, train=False, download=True)
    class_names = [str(c) for c in train_ds.classes]
    _save_split(train_ds, os.path.join(out_root, "train"), class_names, args.resize)
    _save_split(test_ds, os.path.join(out_root, "val"), class_names, args.resize)


def prepare_cifar10(args, out_root: str):
    train_ds = datasets.CIFAR10(args.download_root, train=True, download=True)
    test_ds = datasets.CIFAR10(args.download_root, train=False, download=True)
    class_names = train_ds.classes
    _save_split(train_ds, os.path.join(out_root, "train"), class_names, args.resize)
    _save_split(test_ds, os.path.join(out_root, "val"), class_names, args.resize)


def prepare_svhn(args, out_root: str):
    train_ds = datasets.SVHN(args.download_root, split="train", download=True)
    test_ds = datasets.SVHN(args.download_root, split="test", download=True)
    class_names = [str(i) for i in range(10)]
    _save_split(train_ds, os.path.join(out_root, "train"), class_names, args.resize)
    _save_split(test_ds, os.path.join(out_root, "val"), class_names, args.resize)
    if args.include_svhn_extra:
        extra_ds = datasets.SVHN(args.download_root, split="extra", download=True)
        _save_split(extra_ds, os.path.join(out_root, "extra"), class_names, args.resize)


def prepare_celeba(args, out_root: str):
    try:
        import kagglehub
    except Exception as exc:
        raise RuntimeError(
            "kagglehub is required to download CelebA. "
            "Install it (e.g., `pip install kagglehub`) and ensure Kaggle credentials are set."
        ) from exc

    # Download CelebA from Kaggle via kagglehub.
    dataset_root = kagglehub.dataset_download("jessicali9530/celeba-dataset")

    def find_celeba_assets(root: str):
        partition_path = None
        for dirpath, _, filenames in os.walk(root):
            if "list_eval_partition.txt" in filenames:
                partition_path = os.path.join(dirpath, "list_eval_partition.txt")
                break
            if "list_eval_partition.csv" in filenames:
                partition_path = os.path.join(dirpath, "list_eval_partition.csv")
                break
        if partition_path is None:
            raise FileNotFoundError(
                "Could not find list_eval_partition.txt or list_eval_partition.csv in the Kaggle CelebA download."
            )

        candidates = []
        part_dir = os.path.dirname(partition_path)
        candidates.append(os.path.join(root, "img_align_celeba", "img_align_celeba"))
        candidates.append(os.path.join(root, "img_align_celeba"))
        candidates.append(os.path.join(part_dir, "img_align_celeba", "img_align_celeba"))
        candidates.append(os.path.join(part_dir, "img_align_celeba"))

        for c in candidates:
            if os.path.isdir(c):
                try:
                    sample = next((f for f in os.listdir(c) if f.lower().endswith(".jpg")), None)
                except OSError:
                    sample = None
                if sample:
                    return partition_path, c

        # Fallback: search for a directory containing common CelebA filename.
        target = "000001.jpg"
        for dirpath, _, filenames in os.walk(root):
            if target in filenames:
                return partition_path, dirpath

        raise FileNotFoundError("Could not locate img_align_celeba image directory in the Kaggle CelebA download.")

    partition_path, image_dir = find_celeba_assets(dataset_root)

    splits = {"train": [], "val": [], "test": []}
    if partition_path.endswith(".csv"):
        with open(partition_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                fname, split_id = row[0].strip(), row[1].strip()
                if split_id == "0":
                    splits["train"].append(fname)
                elif split_id == "1":
                    splits["val"].append(fname)
                elif split_id == "2":
                    splits["test"].append(fname)
    else:
        with open(partition_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                fname, split_id = parts[0], parts[1]
                if split_id == "0":
                    splits["train"].append(fname)
                elif split_id == "1":
                    splits["val"].append(fname)
                elif split_id == "2":
                    splits["test"].append(fname)

    if not splits["train"] and not splits["val"] and not splits["test"]:
        raise RuntimeError(
            f"Found partition file at {partition_path}, but could not parse any entries."
        )

    class_names = ["celeba"]

    class ImageListDataset(torch.utils.data.Dataset):
        def __init__(self, root_dir: str, names: List[str], label: int = 0):
            self.root_dir = root_dir
            self.names = names
            self.label = label

        def __len__(self):
            return len(self.names)

        def __getitem__(self, idx):
            img_path = os.path.join(self.root_dir, self.names[idx])
            img = Image.open(img_path)
            return img, self.label

    train_ds = ImageListDataset(image_dir, splits["train"], 0)
    val_ds = ImageListDataset(image_dir, splits["val"], 0)
    test_ds = ImageListDataset(image_dir, splits["test"], 0)

    _save_split(train_ds, os.path.join(out_root, "train"), class_names, args.resize)
    _save_split(val_ds, os.path.join(out_root, "val"), class_names, args.resize)
    _save_split(test_ds, os.path.join(out_root, "test"), class_names, args.resize)


def prepare_imagenet(args, out_root: str):
    if not args.imagenet_root or not os.path.isdir(args.imagenet_root):
        raise ValueError("ImageNet requires --imagenet-root pointing to an extracted ImageNet train/val folder.")
    # torchvision.datasets.ImageFolder already matches the expected layout, so we just sanity check.
    train_dir = os.path.join(args.imagenet_root, "train")
    val_dir = os.path.join(args.imagenet_root, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise ValueError("ImageNet root must contain 'train' and 'val' subfolders in ImageFolder format.")
    print(f"ImageNet already in ImageFolder format at {args.imagenet_root}.")
    print(f"Point train.py --data-path to {train_dir} (or {val_dir}) directly. No copy performed.")
    os.makedirs(out_root, exist_ok=True)


def prepare_tiny_imagenet(args, out_root: str):
    # TinyImageNet from torchvision (since v0.20). If unavailable, instruct manual download.
    try:
        tiny = datasets.TinyImageNet(args.download_root, split="train", download=True)
    except Exception as e:
        raise RuntimeError(
            "TinyImageNet download failed. Ensure torchvision>=0.20 or download manually from "
            "https://www.kaggle.com/c/tiny-imagenet and unpack to download_root."
        ) from e
    val = datasets.TinyImageNet(args.download_root, split="val", download=True)
    class_names = tiny.classes
    _save_split(tiny, os.path.join(out_root, "train"), class_names, args.resize)
    _save_split(val, os.path.join(out_root, "val"), class_names, args.resize)


def main():
    parser = argparse.ArgumentParser(description="Download/prepare datasets into ImageFolder layout.")
    parser.add_argument("--dataset", required=True,
                        choices=["mnist", "cifar10", "svhn", "celeba", "imagenet", "tiny-imagenet"])
    parser.add_argument("--output", type=str, default="data",
                        help="Root output folder where dataset will be written.")
    parser.add_argument("--download-root", type=str, default="raw_data",
                        help="Where to cache raw torchvision downloads.")
    parser.add_argument("--resize", type=int, default=None,
                        help="Optional square resize (e.g., 256) before saving.")
    parser.add_argument("--include-svhn-extra", action="store_true",
                        help="If set, also prepare the SVHN 'extra' split.")
    parser.add_argument("--imagenet-root", type=str, default=None,
                        help="Existing ImageNet root with train/val ImageFolder structure (no auto-download).")
    args = parser.parse_args()

    out_root = os.path.join(args.output, args.dataset)
    os.makedirs(out_root, exist_ok=True)

    if args.dataset == "mnist":
        prepare_mnist(args, out_root)
    elif args.dataset == "cifar10":
        prepare_cifar10(args, out_root)
    elif args.dataset == "svhn":
        prepare_svhn(args, out_root)
    elif args.dataset == "celeba":
        prepare_celeba(args, out_root)
    elif args.dataset == "imagenet":
        prepare_imagenet(args, out_root)
    elif args.dataset == "tiny-imagenet":
        prepare_tiny_imagenet(args, out_root)
    else:
        raise ValueError(f"Unsupported dataset {args.dataset}")

    print(f"Done. Use train.py --data-path {os.path.join(out_root, 'train')} (or val) to start training.")


if __name__ == "__main__":
    main()
