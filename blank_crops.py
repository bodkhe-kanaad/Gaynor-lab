"""
generate_blank_crops.py
=======================
Generates "blank" training crops from guaranteed-empty camera trap images.

Expected directory structure (mirrors the hard drive layout):
    root_dir/
        SiteA or Camera Number/
            Ghost/
                img001.jpg
                img002.jpg
            Ghost2/
                img003.jpg
        SiteB/
            Ghost/
                ...

The site name is inferred from the folder immediately above Ghost/Ghost2. for site based randomization

Usage
-----
Basic (run MD + crop everything above a very smalled threshold intentionally):
    python generate_blank_crops.py \\
        --input_dir  /path/to/empty/images \\
        --output_dir /path/to/blank_crops \\
        --md_version MDV6-yolov10-e

    Model downloads automatically on first run - no .pt file needed. because of the new pytorch based MD

Filter by site:
    python generate_blank_crops.py ... --sites SiteA SiteB

Filter by MD category (animal / person / vehicle):
    python generate_blank_crops.py ... --categories animal

Random subsample instead:
    python generate_blank_crops.py ... --location_sampling random --max_crops 500

Skip re-running MD if you already have a results JSON:
    python generate_blank_crops.py ... --md_json /path/to/md_results.json

Dependencies
------------
    pip install Pillow tqdm

MegaDetector (optional - only needed if not supplying --md_json):
    Follow setup at https://github.com/microsoft/CameraTraps
"""

import os
import sys
import json
import argparse
import random
import math
from pathlib import Path
from collections import defaultdict

from PIL import Image
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

GHOST_FOLDER_NAMES = {"ghost", "ghost2"}   # case-insensitive match
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# MegaDetector category IDs -> human-readable names
MD_CATEGORY_MAP = {
    "1": "animal",
    "2": "person",
    "3": "vehicle",
}


# ──────────────────────────────────────────────────────────────
# Step 1 - Discover images and infer site from directory structure
# ──────────────────────────────────────────────────────────────

def discover_images(root_dir: str, debug: bool = False) -> list:
    """
    Walk root_dir and find all images inside Ghost / Ghost2 subfolders.
    Returns a list of dicts:
        {
            "path":   Path,   # absolute path to image file
            "site":   str,    # name of the folder above Ghost/Ghost2
            "camera": str,    # "Ghost" or "Ghost2"
        }

    The expected structure is:
        root_dir / <site> / Ghost[2] / image.jpg

    If an image is directly inside a Ghost folder with no parent site
    folder, site is set to "unknown".
    """
    root = Path(root_dir).resolve()
    found = []

    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            if debug:
                print(f"  skip (extension): {path.name}")
            continue

        # Walk up the relative path to find a Ghost/Ghost2 ancestor
        # Structure can be either:
        #   site/Ghost/image.jpg
        #   site/Ghost/subfolder/image.jpg  (e.g. Ghost/1/image.jpg)
        parts = path.relative_to(root).parts
        ghost_idx = None
        for i, part in enumerate(parts[:-1]):   # exclude filename itself
            if part.lower() in GHOST_FOLDER_NAMES:
                ghost_idx = i
                break

        if ghost_idx is None:
            if debug:
                print(f"  skip (no Ghost ancestor): {parts}")
            continue

        camera = parts[ghost_idx]                           # "Ghost" or "Ghost2"
        site   = parts[ghost_idx - 1] if ghost_idx > 0 else "unknown"

        if debug:
            print(f"  found: site={site!r} camera={camera!r} file={path.name}")

        found.append({
            "path":   path,
            "site":   site,
            "camera": camera,
        })

    return found


# ──────────────────────────────────────────────────────────────
# Step 2 - Run MegaDetector (or load existing results)
# ──────────────────────────────────────────────────────────────

def run_megadetector(image_records: list, md_version: str = "MDV6-yolov10-e",
                     threshold: float = 0.05, batch_size: int = 16) -> dict:
    """
    Runs MegaDetector (via PytorchWildlife) on all images and returns a
    results dict keyed by absolute image path string.

    Uses the new PytorchWildlife API — no .pt file needed, the model is
    downloaded automatically on first run and cached locally.

    Valid md_version values:
        MDV6-yolov9-c  MDV6-yolov9-e
        MDV6-yolov10-c MDV6-yolov10-e  (default, best accuracy)
        MDV6-rtdetr-c
        MV5 (uses MegaDetectorV5 instead)

    We run MD at a low threshold (default 0.05) to capture everything -
    filtering by the user's chosen --threshold happens later, so you
    don't have to re-run MD just to try a different cutoff.

    Returns:
        {
            "/abs/path/to/img.jpg": [
                {"bbox": [x, y, w, h],  # relative coords 0-1
                 "conf": 0.83,
                 "category": "1"},      # "1"=animal "2"=person "3"=vehicle
                ...
            ],
            ...
        }
    """
    try:
        import torch
        import sys as _sys
        _sys.path.insert(0, os.path.expanduser("~/CameraTraps"))
        from PytorchWildlife.models import detection as pw_detection
    except ImportError:
        raise RuntimeError(
            "PytorchWildlife not found. Install it with:\n"
            "  pip install PytorchWildlife\n"
            "Or supply a pre-run results file with --md_json"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Load the model - downloads automatically on first run
    if md_version == "MV5":
        print(f"  Loading MegaDetectorV5 (version a) ...")
        detector = pw_detection.MegaDetectorV5(device=device, pretrained=True, version="a")
    else:
        print(f"  Loading MegaDetectorV6 (version {md_version}) ...")
        detector = pw_detection.MegaDetectorV6(device=device, pretrained=True, version=md_version)

    # PytorchWildlife batch detection works on a folder path, so we
    # collect all unique parent folders and run per-folder, then index
    # results back to our image records by path.
    #
    # det_results is a list of dicts, one per image:
    #   {
    #     "img_id": "filename.jpg",
    #     "detections": sv.Detections object  (xyxy pixel coords + confidence + class_id)
    #   }

    all_paths   = [r["path"] for r in image_records]
    path_set    = {str(p) for p in all_paths}

    # Get unique folders so we can run batch detection per folder
    folders = sorted({str(p.parent) for p in all_paths})

    raw_by_path = {}  # str(path) -> list of raw detection dicts

    for folder in folders:
        print(f"  Running MD on folder: {folder}")
        try:
            det_results = detector.batch_image_detection(folder, batch_size=batch_size)
        except Exception as e:
            print(f"  Warning: batch detection failed on {folder}: {e}")
            continue

        for det in det_results:
            # img_id is just the filename - reconstruct full path
            img_filename = det.get("img_id", "")
            full_path    = str(Path(folder) / img_filename)

            if full_path not in path_set:
                continue  # image wasn't in our Ghost folder list

            detections_sv = det.get("detections")
            if detections_sv is None or len(detections_sv) == 0:
                raw_by_path[full_path] = []
                continue

            # sv.Detections stores boxes as xyxy in pixel coords.
            # We need to convert to relative [x, y, w, h] (MD standard format).
            try:
                img_w, img_h = Image.open(full_path).size
            except Exception:
                raw_by_path[full_path] = []
                continue

            det_list = []
            for i, xyxy in enumerate(detections_sv.xyxy):
                conf     = float(detections_sv.confidence[i])
                class_id = int(detections_sv.class_id[i])

                if conf < threshold:
                    continue

                x1, y1, x2, y2 = xyxy
                rel_x = x1 / img_w
                rel_y = y1 / img_h
                rel_w = (x2 - x1) / img_w
                rel_h = (y2 - y1) / img_h

                # PytorchWildlife class_id: 0=animal, 1=person, 2=vehicle
                # Map to MD category strings: "1"=animal, "2"=person, "3"=vehicle
                category = str(class_id + 1)

                det_list.append({
                    "bbox":     [rel_x, rel_y, rel_w, rel_h],
                    "conf":     conf,
                    "category": category,
                })

            raw_by_path[full_path] = det_list

    # Fill in empty results for any images MD didn't return
    for record in image_records:
        key = str(record["path"])
        if key not in raw_by_path:
            raw_by_path[key] = []

    return raw_by_path


def load_md_json(json_path: str, threshold: float) -> dict:
    """
    Load a MegaDetector results JSON (output of run_detector_batch.py).
    Filters to detections at or above threshold.
    """
    print(f"Loading MD results from {json_path} ...")
    with open(json_path) as f:
        data = json.load(f)

    results = {}
    for entry in data.get("images", []):
        key = entry["file"]
        results[key] = [
            {
                "bbox":     d["bbox"],
                "conf":     d["conf"],
                "category": str(d.get("category", "1")),
            }
            for d in (entry.get("detections") or [])
            if d["conf"] >= threshold
        ]
    return results


def save_md_json(results: dict, output_dir: str):
# ToDo


# ──────────────────────────────────────────────────────────────
# Step 3 - Build detection list with metadata and apply filters
# ──────────────────────────────────────────────────────────────

def build_detection_list(image_records: list, md_results: dict,
                          threshold: float,
                          categories: list,
                          sites: list) -> list:
    """
    Merges image metadata with MD detections, then applies filters:
        - confidence threshold
        - MD category (animal / person / vehicle)
        - site name

    Returns a flat list of dicts, each containing everything needed
    for cropping and filename generation:
        {
            "path":     Path,
            "site":     str,
            "camera":   str,
            "bbox":     [x, y, w, h],  # relative coords
            "conf":     float,
            "category": str,           # "animal" / "person" / "vehicle"
        }
    """
    allowed_categories = (
        {c.lower() for c in categories} if categories else None
    )
    allowed_sites = (
        {s.lower() for s in sites} if sites else None
    )

    detections = []

    for record in image_records:
        path = record["path"]
        site = record["site"]

        # Site filter
        if allowed_sites and site.lower() not in allowed_sites:
            continue

        raw_dets = md_results.get(str(path), [])
        for d in raw_dets:
            conf     = d["conf"]
            cat_id   = d["category"]
            cat_name = MD_CATEGORY_MAP.get(cat_id, "unknown")

            # Confidence filter
            if conf < threshold:
                continue

            # Category filter
            if allowed_categories and cat_name.lower() not in allowed_categories:
                continue

            detections.append({
                "path":     path,
                "site":     site,
                "camera":   record["camera"],
                "bbox":     d["bbox"],
                "conf":     conf,
                "category": cat_name,
            })

    return detections


# ──────────────────────────────────────────────────────────────
# Step 4 - Subsample by pixel location
# ──────────────────────────────────────────────────────────────

def get_grid_cell(bbox: list, n_cells: int = 9) -> int:
 # ToDo


def subsample_by_location(detections: list, strategy: str,
                           max_crops: int,
                           grid_cells: int = 9) -> list:
    """
    Subsample detections to reduce spatial redundancy.

    strategy="grid":
        Groups detections by (site, camera, grid_cell) and round-robins
        across groups, taking the highest-confidence detection from each
        group first. This ensures every site, camera, and frame region
        contributes roughly equally.

    strategy="random":
        Shuffle and truncate to max_crops. Simple but less principled.

    strategy="none":
        No subsampling - return everything sorted by confidence descending.
    """
    if strategy == "none":
        detections.sort(key=lambda d: d["conf"], reverse=True)
        if max_crops:
            detections = detections[:max_crops]
        return detections

    if strategy == "random":
        random.shuffle(detections)
        if max_crops:
            detections = detections[:max_crops]
        return detections

    if strategy == "grid":
        # Group by (site, camera, grid_cell)
        buckets = defaultdict(list)
        for d in detections:
            cell = get_grid_cell(d["bbox"], grid_cells)
            key  = (d["site"], d["camera"], cell)
            buckets[key].append(d)

        # Sort each bucket by confidence descending
        for key in buckets:
            buckets[key].sort(key=lambda d: d["conf"], reverse=True)

        # Round-robin across all buckets
        bucket_keys = sorted(buckets.keys())
        pointers    = {k: 0 for k in bucket_keys}
        selected    = []

        while True:
            added = 0
            for key in bucket_keys:
                if max_crops and len(selected) >= max_crops:
                    break
                idx = pointers[key]
                if idx < len(buckets[key]):
                    selected.append(buckets[key][idx])
                    pointers[key] += 1
                    added += 1
            if added == 0:
                break

        return selected

    raise ValueError(f"Unknown location_sampling strategy: {strategy!r}")


# ──────────────────────────────────────────────────────────────
# Step 5 - Crop and save
# ──────────────────────────────────────────────────────────────

def bbox_to_pixels(bbox: list, img_w: int, img_h: int) -> tuple:
    """Convert MD relative bbox [x, y, w, h] to pixel (left, top, right, bottom)."""
    x, y, w, h = bbox
    left   = max(0, int(x * img_w))
    top    = max(0, int(y * img_h))
    right  = min(img_w, int((x + w) * img_w))
    bottom = min(img_h, int((y + h) * img_h))
    return left, top, right, bottom


def make_filename(detection: dict, crop_idx: int) -> str:
    """
    Confidence-first filename so file viewers sort highest-confidence first.
    Example: blank_conf_0p813_site_SiteA_camera_Ghost_animal_crop_0001.jpg
    """
    conf_str   = f"{detection['conf']:.3f}".replace(".", "p")
    site_str   = detection["site"].replace(" ", "_")
    camera_str = detection["camera"].replace(" ", "_")
    cat_str    = detection["category"]
    return (
        f"blank_conf_{conf_str}"
        f"_site_{site_str}"
        f"_camera_{camera_str}"
        f"_{cat_str}"
        f"_crop_{crop_idx:04d}.jpg"
    )


def crop_and_save(detection: dict, crop_idx: int,
                  output_dir: Path, padding: int) -> Path:
    """Crop the detection box (plus padding) from the image and save as JPEG."""
    img = Image.open(detection["path"]).convert("RGB")
    w, h = img.size

    left, top, right, bottom = bbox_to_pixels(detection["bbox"], w, h)

    # Apply padding, clamped to image bounds
    left   = max(0, left   - padding)
    top    = max(0, top    - padding)
    right  = min(w, right  + padding)
    bottom = min(h, bottom + padding)

    crop     = img.crop((left, top, right, bottom))
    filename = make_filename(detection, crop_idx)
    out_path = output_dir / filename
    crop.save(out_path, "JPEG", quality=90)

    return out_path


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate blank training crops from guaranteed-empty camera trap images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    parser.add_argument("--input_dir",  required=True,
        help="Root folder containing site subfolders with Ghost/Ghost2 directories.")
    parser.add_argument("--output_dir", required=True,
        help="Where to write the blank crops.")

    # MegaDetector source - one of these is required
    md_group = parser.add_mutually_exclusive_group(required=True)
    md_group.add_argument("--md_version",
        default=None,
        help="Run MegaDetector via PytorchWildlife. Model downloads automatically. "
             "Options: MDV6-yolov10-e (default/best), MDV6-yolov10-c, "
             "MDV6-yolov9-e, MDV6-yolov9-c, MDV6-rtdetr-c, MV5. "
             "Example: --md_version MDV6-yolov10-e")
    md_group.add_argument("--md_json",
        help="Path to a pre-run MD results JSON. Skips running MD entirely.")

    # Filtering
    parser.add_argument("--threshold", type=float, default=0.1,
        help="Minimum MD confidence to keep a detection. (default: 0.1)")
    parser.add_argument("--sites", nargs="+", default=None,
        help="Only include images from these site names. "
             "Default: all sites. Example: --sites SiteA SiteB")
    parser.add_argument("--categories", nargs="+",
        choices=["animal", "person", "vehicle"], default=None,
        help="Only keep detections of these MD categories. "
             "Default: all. Example: --categories animal")

    # Spatial subsampling
    parser.add_argument("--location_sampling",
        choices=["grid", "random", "none"], default="grid",
        help="How to subsample by pixel location.\n"
             "  grid   - round-robin across frame regions (recommended)\n"
             "  random - random shuffle then truncate\n"
             "  none   - keep everything above threshold\n"
             "(default: grid)")
    parser.add_argument("--grid_cells", type=int, default=9,
        help="Number of grid cells for --location_sampling grid. "
             "Must be a perfect square: 4, 9, 16. (default: 9)")
    parser.add_argument("--max_crops", type=int, default=None,
        help="Maximum total blank crops to generate. "
             "Aim for the same order of magnitude as your largest species class.")

    # Crop appearance
    parser.add_argument("--padding", type=int, default=32,
        help="Pixels of context padding around each detection box. (default: 32)")

    # MD run options
    parser.add_argument("--md_threshold", type=float, default=0.05,
        help="Confidence floor passed to MD at inference time. "
             "Keep this lower than --threshold so you can adjust filtering "
             "later without re-running MD. (default: 0.05)")
    parser.add_argument("--batch_size", type=int, default=16,
        help="Batch size for MegaDetector inference. (default: 16)")
    parser.add_argument("--save_md_json", action="store_true",
        help="Save the MD results JSON to output_dir for reuse later.")

    parser.add_argument("--debug", action="store_true",
        help="Print every file the scanner considers, to diagnose path issues.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover images
    print(f"\n[1/5] Scanning for Ghost/Ghost2 images in: {args.input_dir}")
    image_records = discover_images(args.input_dir, debug=args.debug)

    if not image_records:
        print("  No images found. Check --input_dir and that folders are named Ghost or Ghost2.")
        sys.exit(1)

    sites_found = sorted({r["site"] for r in image_records})
    print(f"  Found {len(image_records)} images across {len(sites_found)} sites: {sites_found}")

    # 2. Run MD or load existing results
    print(f"\n[2/5] Getting MegaDetector results ...")
    if args.md_json:
        md_results = load_md_json(args.md_json, threshold=args.md_threshold)
    else:
        md_results = run_megadetector(
            image_records,
            md_version=args.md_version or "MDV6-yolov10-e",
            threshold=args.md_threshold,
            batch_size=args.batch_size,
        )
        if args.save_md_json:
            save_md_json(md_results, args.output_dir)

    # 3. Build and filter detection list
    print(f"\n[3/5] Filtering detections ...")
    print(f"  Confidence threshold : {args.threshold}")
    print(f"  Sites filter         : {args.sites or 'all'}")
    print(f"  Category filter      : {args.categories or 'all'}")

    detections = build_detection_list(
        image_records, md_results,
        threshold=args.threshold,
        categories=args.categories,
        sites=args.sites,
    )
    print(f"  Detections after filtering: {len(detections)}")

    if not detections:
        print("  No detections passed the filters. Try lowering --threshold.")
        sys.exit(0)

    # 4. Subsample by location
    print(f"\n[4/5] Subsampling (strategy={args.location_sampling}) ...")
    detections = subsample_by_location(
        detections,
        strategy=args.location_sampling,
        max_crops=args.max_crops,
        grid_cells=args.grid_cells,
    )
    print(f"  Crops to generate: {len(detections)}")

    # 5. Crop and save
    print(f"\n[5/5] Saving crops to: {output_dir}")
    failed   = 0
    per_site = defaultdict(int)

    for i, det in enumerate(tqdm(detections)):
        try:
            crop_and_save(det, crop_idx=i + 1,
                          output_dir=output_dir, padding=args.padding)
            per_site[det["site"]] += 1
        except Exception as e:
            print(f"  Warning: failed on {det['path'].name}: {e}")
            failed += 1

    saved = len(detections) - failed
    print(f"\nDone. Saved {saved} blank crops ({failed} failed).")
    print(f"\nCrops per site:")
    for site, count in sorted(per_site.items()):
        print(f"  {site}: {count}")
    print(f"\nNext step:")
    print(f"  Open {output_dir} in IrfanView (or similar fast viewer).")
    print(f"  Files sort by confidence - review highest-confidence crops first.")
    print(f"  Delete anything that turns out to contain a real animal.")
    print(f"  Then use this folder as your 'blank' class in training.")


if __name__ == "__main__":
    main()