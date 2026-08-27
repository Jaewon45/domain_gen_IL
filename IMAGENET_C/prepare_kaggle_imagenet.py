#!/usr/bin/env python3
"""Prepare a torchvision-compatible ImageNet val view from Kaggle localization data.

This script never mutates the source Kaggle directory.
"""

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


EXPECTED_VAL_IMAGES = 50000
EXPECTED_NUM_CLASSES = 1000
IMAGE_ID_PATTERN = re.compile(r"^ILSVRC2012_val_(\d{8})$")


def resolve_default_paths(kaggle_root: Path) -> Tuple[Path, Path, Path]:
    val_dir = kaggle_root / "ILSVRC" / "Data" / "CLS-LOC" / "val"
    val_solution = kaggle_root / "LOC_val_solution.csv"
    synset_mapping = kaggle_root / "LOC_synset_mapping.txt"
    return val_dir, val_solution, synset_mapping


def parse_prediction_string(prediction: str) -> str:
    tokens = prediction.strip().split()
    if not tokens:
        raise ValueError("PredictionString is empty")
    if len(tokens) % 5 != 0:
        raise ValueError(f"PredictionString has invalid token count: {len(tokens)}")

    wnids = [tokens[i] for i in range(0, len(tokens), 5)]
    first = wnids[0]
    if any(wnid != first for wnid in wnids):
        raise ValueError(f"PredictionString contains mixed WNIDs: {wnids}")
    return first


def image_id_sort_key(image_id: str) -> int:
    match = IMAGE_ID_PATTERN.match(image_id)
    if not match:
        raise ValueError(f"Invalid ImageId format: {image_id}")
    return int(match.group(1))


def parse_val_solution_csv(path: Path) -> Tuple[Dict[str, str], List[str]]:
    image_to_wnid: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ImageId", "PredictionString"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV missing required columns {required}: {reader.fieldnames}")

        for row in reader:
            image_id = (row.get("ImageId") or "").strip()
            pred = row.get("PredictionString") or ""
            if not image_id:
                raise ValueError("Found empty ImageId in LOC_val_solution.csv")
            if image_id in image_to_wnid:
                raise ValueError(f"Duplicate ImageId in LOC_val_solution.csv: {image_id}")
            image_to_wnid[image_id] = parse_prediction_string(pred)

    ordered_image_ids = sorted(image_to_wnid.keys(), key=image_id_sort_key)
    return image_to_wnid, ordered_image_ids


def parse_synset_mapping(path: Path) -> Dict[str, Tuple[str, ...]]:
    wnid_to_classes: Dict[str, Tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid synset mapping line {line_num}: {raw_line.rstrip()}")
            wnid, description = parts
            class_names = tuple(part.strip() for part in description.split(",") if part.strip())
            if not class_names:
                raise ValueError(f"No class names parsed at line {line_num}: {raw_line.rstrip()}")
            if wnid in wnid_to_classes:
                raise ValueError(f"Duplicate WNID in synset mapping: {wnid}")
            wnid_to_classes[wnid] = class_names
    return wnid_to_classes


def list_jpeg_image_ids(val_dir: Path) -> Dict[str, Path]:
    files = sorted(val_dir.glob("*.JPEG"))
    mapping: Dict[str, Path] = {}
    for file_path in files:
        image_id = file_path.stem
        if image_id in mapping:
            raise ValueError(f"Duplicate JPEG stem in val directory: {image_id}")
        mapping[image_id] = file_path
    return mapping


def ensure_expected_counts(
    image_to_wnid: Dict[str, str],
    source_images: Dict[str, Path],
    expected_rows: int,
    expected_classes: int,
) -> None:
    if len(image_to_wnid) != expected_rows:
        raise ValueError(f"Validation CSV row count mismatch: {len(image_to_wnid)} != {expected_rows}")
    if len(source_images) != expected_rows:
        raise ValueError(f"Validation JPEG file count mismatch: {len(source_images)} != {expected_rows}")

    missing_images = sorted(set(image_to_wnid.keys()) - set(source_images.keys()))
    extra_images = sorted(set(source_images.keys()) - set(image_to_wnid.keys()))
    if missing_images:
        raise FileNotFoundError(f"Missing validation JPEG files for {len(missing_images)} ids; first: {missing_images[0]}")
    if extra_images:
        raise ValueError(f"Found {len(extra_images)} extra validation JPEG files not in CSV; first: {extra_images[0]}")

    unique_wnids = set(image_to_wnid.values())
    if len(unique_wnids) != expected_classes:
        raise ValueError(f"Validation unique WNID count mismatch: {len(unique_wnids)} != {expected_classes}")


def validate_wnid_mapping(
    image_to_wnid: Dict[str, str],
    wnid_to_classes: Dict[str, Tuple[str, ...]],
    expected_classes: int,
) -> None:
    if len(wnid_to_classes) != expected_classes:
        raise ValueError(f"Synset mapping WNID count mismatch: {len(wnid_to_classes)} != {expected_classes}")
    unknown = sorted(set(image_to_wnid.values()) - set(wnid_to_classes.keys()))
    if unknown:
        raise ValueError(f"Validation CSV references unknown WNIDs; first: {unknown[0]}")


def same_volume(path_a: Path, path_b: Path) -> bool:
    drive_a = os.path.splitdrive(str(path_a.resolve()))[0].lower()
    drive_b = os.path.splitdrive(str(path_b.resolve()))[0].lower()
    return drive_a == drive_b


def files_match(src: Path, dst: Path) -> bool:
    if not dst.exists() or not src.exists():
        return False
    try:
        if os.path.samefile(src, dst):
            return True
    except OSError:
        pass
    if src.stat().st_size != dst.stat().st_size:
        return False
    return filecmp_bytes(src, dst)


def filecmp_bytes(path_a: Path, path_b: Path, chunk_size: int = 1024 * 1024) -> bool:
    with path_a.open("rb") as fa, path_b.open("rb") as fb:
        while True:
            a = fa.read(chunk_size)
            b = fb.read(chunk_size)
            if a != b:
                return False
            if not a:
                return True


def link_or_copy_file(src: Path, dst: Path, requested_mode: str, overwrite: bool) -> str:
    if dst.exists():
        if files_match(src, dst):
            return "skip"
        if not overwrite:
            raise FileExistsError(f"Destination exists with mismatched content: {dst}")
        if dst.is_dir():
            raise IsADirectoryError(f"Destination path is a directory: {dst}")
        dst.unlink()

    if requested_mode == "copy":
        shutil.copy2(src, dst)
        return "copy"

    if requested_mode == "symlink":
        os.symlink(src, dst)
        return "symlink"

    if requested_mode == "hardlink":
        if not same_volume(src, dst.parent):
            raise OSError(f"Hardlink requested but source and destination are on different volumes: {src} -> {dst}")
        os.link(src, dst)
        return "hardlink"

    if requested_mode == "auto":
        if same_volume(src, dst.parent):
            try:
                os.link(src, dst)
                return "hardlink"
            except OSError:
                pass
        shutil.copy2(src, dst)
        return "copy"

    raise ValueError(f"Unsupported link mode: {requested_mode}")


def ensure_val_class_dirs(output_root: Path, wnids: Sequence[str], dry_run: bool) -> None:
    if dry_run:
        return
    val_root = output_root / "val"
    val_root.mkdir(parents=True, exist_ok=True)
    for wnid in wnids:
        (val_root / wnid).mkdir(parents=True, exist_ok=True)


def build_manifest_rows(
    ordered_image_ids: Sequence[str],
    image_to_wnid: Dict[str, str],
    source_images: Dict[str, Path],
    output_root: Path,
) -> Iterable[Tuple[str, str, Path, Path]]:
    val_root = output_root / "val"
    for image_id in ordered_image_ids:
        wnid = image_to_wnid[image_id]
        src = source_images[image_id]
        dst = val_root / wnid / f"{image_id}.JPEG"
        yield image_id, wnid, src, dst


def write_prepare_manifest(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "wnid", "source_path", "destination_path", "link_mode"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_meta_bin(output_root: Path, wnid_to_classes: Dict[str, Tuple[str, ...]], val_wnids: Sequence[str], dry_run: bool) -> bool:
    if dry_run:
        return False
    meta_path = output_root / "meta.bin"
    output_root.mkdir(parents=True, exist_ok=True)
    torch.save((wnid_to_classes, list(val_wnids)), meta_path)
    return True


def validate_prepared_imagenet_root(imagenet_root: Path) -> Dict[str, int]:
    from torchvision.datasets import ImageNet  # type: ignore

    clean = ImageNet(root=str(imagenet_root), split="val")
    val_root = imagenet_root / "val"
    val_dirs = [entry for entry in val_root.iterdir() if entry.is_dir()]
    val_images = list(val_root.glob("*/*.JPEG"))

    return {
        "len_dataset": len(clean),
        "len_wnids": len(clean.wnids),
        "len_wnid_to_idx": len(clean.wnid_to_idx),
        "val_dir_count": len(val_dirs),
        "val_image_count": len(val_images),
    }


def summarize_mode(mode_counts: Dict[str, int]) -> str:
    used = [mode for mode, count in mode_counts.items() if count > 0 and mode != "skip"]
    if not used:
        return "skip"
    if len(used) == 1:
        return used[0]
    return "mixed:" + ",".join(sorted(used))


def run(args: argparse.Namespace) -> None:
    kaggle_root = Path(args.kaggle_root).resolve()
    output_root = Path(args.output_root).resolve()

    default_val_dir, default_solution, default_mapping = resolve_default_paths(kaggle_root)
    val_dir = Path(args.val_dir).resolve() if args.val_dir else default_val_dir
    val_solution = Path(args.val_solution).resolve() if args.val_solution else default_solution
    synset_mapping = Path(args.synset_mapping).resolve() if args.synset_mapping else default_mapping

    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation image directory not found: {val_dir}")
    if not val_solution.is_file():
        raise FileNotFoundError(f"LOC_val_solution.csv not found: {val_solution}")
    if not synset_mapping.is_file():
        raise FileNotFoundError(f"LOC_synset_mapping.txt not found: {synset_mapping}")

    image_to_wnid, ordered_image_ids = parse_val_solution_csv(val_solution)
    source_images = list_jpeg_image_ids(val_dir)
    wnid_to_classes = parse_synset_mapping(synset_mapping)

    ensure_expected_counts(image_to_wnid, source_images, EXPECTED_VAL_IMAGES, EXPECTED_NUM_CLASSES)
    validate_wnid_mapping(image_to_wnid, wnid_to_classes, EXPECTED_NUM_CLASSES)

    val_wnids = [image_to_wnid[image_id] for image_id in ordered_image_ids]
    all_wnids_sorted = sorted(wnid_to_classes.keys())

    print(f"Validation labels: {len(image_to_wnid)}")
    print(f"Validation JPEGs: {len(source_images)}")
    print(f"Unique WNIDs: {len(set(val_wnids))}")
    print("Source dataset will not be modified")
    print(f"Requested link mode: {args.link_mode}")

    ensure_val_class_dirs(output_root, all_wnids_sorted, args.dry_run)

    mode_counts: Dict[str, int] = {"hardlink": 0, "copy": 0, "symlink": 0, "skip": 0}
    manifest_rows: List[Dict[str, str]] = []

    for image_id, wnid, src, dst in build_manifest_rows(ordered_image_ids, image_to_wnid, source_images, output_root):
        action = "skip" if args.dry_run else link_or_copy_file(src, dst, args.link_mode, args.overwrite)
        if action not in mode_counts:
            mode_counts[action] = 0
        mode_counts[action] += 1
        manifest_rows.append(
            {
                "image_id": image_id,
                "wnid": wnid,
                "source_path": str(src),
                "destination_path": str(dst),
                "link_mode": action,
            }
        )

    meta_bin_created = write_meta_bin(output_root, wnid_to_classes, val_wnids, args.dry_run)

    if not args.dry_run:
        write_prepare_manifest(output_root / "prepare_manifest.csv", manifest_rows)

    mode_used = summarize_mode(mode_counts)

    summary = {
        "source_root": str(kaggle_root),
        "output_root": str(output_root),
        "num_images": len(image_to_wnid),
        "num_classes": len(set(val_wnids)),
        "link_mode_requested": args.link_mode,
        "link_mode_used": mode_used,
        "meta_bin_created": meta_bin_created,
        "source_modified": False,
        "dry_run": bool(args.dry_run),
    }

    if not args.dry_run:
        with (output_root / "prepare_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    print(f"Link mode used: {mode_used}")
    print(f"Skipped existing files: {mode_counts.get('skip', 0)}")

    if args.verify:
        if args.dry_run:
            print("--verify requested with --dry_run; skipping loader validation.")
        else:
            stats = validate_prepared_imagenet_root(output_root)
            print(f"Verify ImageNet len: {stats['len_dataset']}")
            print(f"Verify wnids: {stats['len_wnids']}")
            print(f"Verify wnid_to_idx: {stats['len_wnid_to_idx']}")
            print(f"Verify val dirs: {stats['val_dir_count']}")
            print(f"Verify val images: {stats['val_image_count']}")

            if stats["len_dataset"] != EXPECTED_VAL_IMAGES:
                raise ValueError(f"Prepared clean ImageNet image count mismatch: {stats['len_dataset']}")
            if stats["len_wnids"] != EXPECTED_NUM_CLASSES:
                raise ValueError(f"Prepared clean ImageNet wnid count mismatch: {stats['len_wnids']}")
            if stats["len_wnid_to_idx"] != EXPECTED_NUM_CLASSES:
                raise ValueError(f"Prepared clean ImageNet wnid_to_idx mismatch: {stats['len_wnid_to_idx']}")
            if stats["val_dir_count"] != EXPECTED_NUM_CLASSES:
                raise ValueError(f"Prepared val directory count mismatch: {stats['val_dir_count']}")
            if stats["val_image_count"] != EXPECTED_VAL_IMAGES:
                raise ValueError(f"Prepared val image count mismatch on disk: {stats['val_image_count']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Kaggle ImageNet val for torchvision.datasets.ImageNet")
    parser.add_argument("--kaggle_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--link_mode", choices=["auto", "hardlink", "copy", "symlink"], default="auto")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--val_dir", default=None)
    parser.add_argument("--val_solution", default=None)
    parser.add_argument("--synset_mapping", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
