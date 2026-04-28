"""
test_review_crops.py
====================
Generates a minimal synthetic test environment for review_crops.py.
 
Creates:
  - A few fake source images with a coloured rectangle (the "animal")
  - A full_df_filtered.csv with the correct columns (filename, species,
    site, bboxes, filename_crops, det_confs, det_categories)
  - Runs Step 1 (generate review crops with 100px padding)
  - Prints instructions for Step 2
 
Run:
    python test_review_crops.py
"""
 
import json
import os
import shutil
from pathlib import Path
 
import pandas as pd
from PIL import Image, ImageDraw
 
# ──────────────────────────────────────────────────────────────
# Config — change these paths if needed
# ──────────────────────────────────────────────────────────────
 
BASE          = Path("/Users/bodkhe.kanaad/Gaynor Lab/review_crops_test")
IMAGE_DIR     = BASE / "source_images"    # fake original images go here
CROPS_DIR     = BASE / "crops"            # where final re-crops will go
REVIEW_DIR    = BASE / "review_crops"     # padded review crops go here
FILTERED_CSV  = BASE / "full_df_filtered.csv"
OUTPUT_CSV    = BASE / "full_df_filtered_reviewed.csv"
 
 
# ──────────────────────────────────────────────────────────────
# Synthetic test data
# ──────────────────────────────────────────────────────────────
 
# Each entry: (filename, species, site, bbox_relative [x,y,w,h])
# We put a bright coloured rectangle at the bbox location
# so you can visually confirm the crop contains the "animal"
TEST_CASES = [
    ("zebra_A06_001.jpg",  "zebra",   "a06", [0.3,  0.3,  0.2, 0.3]),
    ("zebra_A06_002.jpg",  "zebra",   "a06", [0.5,  0.1,  0.3, 0.4]),
    ("baboon_A10_001.jpg", "baboon",  "a10", [0.1,  0.5,  0.25, 0.25]),
    ("baboon_A10_002.jpg", "baboon",  "a10", [0.6,  0.6,  0.2, 0.2]),
    # One image with two detections (tests multi-crop handling)
    ("lion_A06_001.jpg",   "lion",    "a06", [0.1, 0.1, 0.2, 0.2]),
]
 
# Second detection for the multi-crop image
SECOND_DET = ("lion_A06_001.jpg", [0.65, 0.55, 0.2, 0.2])
 
 
def make_test_image(path: Path, bbox: list, size=(800, 600)):
    """Create a grey image with a bright rectangle at bbox location."""
    img  = Image.new("RGB", size, color=(180, 180, 180))
    draw = ImageDraw.Draw(img)
 
    W, H = size
    x, y, w, h = bbox
    x1 = int(x * W);  y1 = int(y * H)
    x2 = int((x+w)*W); y2 = int((y+h)*H)
 
    # Fill the "animal" area in bright orange
    draw.rectangle([x1, y1, x2, y2], fill=(230, 120, 30))
    # Draw a border so it's obvious in review
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
 
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)
 
 
def build_test_environment():
    print(f"Building test environment in: {BASE}")
    BASE.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
 
    rows = []
    crop_idx_by_file = {}
 
    for filename, species, site, bbox in TEST_CASES:
        img_path = IMAGE_DIR / filename
        make_test_image(img_path, bbox)
 
        # Assign crop indices (some images get two crops)
        idx = crop_idx_by_file.get(filename, 0)
        stem = Path(filename).stem
 
        bboxes      = [bbox]
        crop_names  = [f"{stem}_crop{idx}.jpg"]
        det_confs   = [0.85]
        det_cats    = ["1"]
 
        # Add second detection for lion image
        if filename == "lion_A06_001.jpg":
            bbox2 = SECOND_DET[1]
            bboxes.append(bbox2)
            crop_names.append(f"{stem}_crop1.jpg")
            det_confs.append(0.72)
            det_cats.append("1")
 
        crop_idx_by_file[filename] = idx + len(crop_names)
 
        rows.append({
            "filename":        filename,
            "species":         species,
            "site":            site,
            "n_animals":       len(bboxes),
            "filename_crops":  json.dumps(crop_names),
            "bboxes":          json.dumps(bboxes),
            "det_confs":       json.dumps(det_confs),
            "det_categories":  json.dumps(det_cats),
            "filename_crop":   crop_names[0],
            "bbox":            json.dumps(bboxes[0]),
            "det_conf":        det_confs[0],
            "det_category":    det_cats[0],
        })
 
    df = pd.DataFrame(rows)
    df.to_csv(FILTERED_CSV, index=False)
 
    print(f"  Created {len(rows)} source images in:  {IMAGE_DIR}")
    print(f"  Created CSV ({len(df)} rows) at:       {FILTERED_CSV}")
    print()
 
    # Count expected crops
    total_crops = sum(
        len(json.loads(r["filename_crops"])) for _, r in df.iterrows()
    )
    print(f"  Expected total crops: {total_crops}")
    print(f"    (lion_A06_001 has 2 detections, all others have 1)")
    return df
 
 
def run_step1():
    print("\n" + "="*60)
    print("RUNNING STEP 1 — generate review crops")
    print("="*60)
    os.system(
        f'python "/Users/bodkhe.kanaad/Gaynor Lab/Gaynor-lab/review_crops.py" generate '
        f'--filtered_csv "{FILTERED_CSV}" '
        f'--image_dir    "{IMAGE_DIR}" '
        f'--review_dir   "{REVIEW_DIR}"'
    )
 
 
def print_step2_instructions():
    print("\n" + "="*60)
    print("STEP 1 COMPLETE — now do the manual review:")
    print("="*60)
    print(f"""
1. Open this folder in Finder:
   {REVIEW_DIR}
 
2. You'll see subfolders by species: baboon/, lion/, zebra/
   Each crop has 100px of context around the orange rectangle.
 
3. Delete one or two crops to simulate rejecting bad images.
   For example, delete:
   {REVIEW_DIR}/zebra/zebra_A06_001_crop0.jpg
 
4. Then run Step 2:
 
   python "/Users/bodkhe.kanaad/Gaynor Lab/Gaynor-lab/review_crops.py" recrop \\
       --filtered_csv  "{FILTERED_CSV}" \\
       --image_dir     "{IMAGE_DIR}" \\
       --review_dir    "{REVIEW_DIR}" \\
       --output_dir    "{CROPS_DIR}" \\
       --output_csv    "{OUTPUT_CSV}"
 
5. Check the results:
   - {CROPS_DIR}  should contain final crops WITHOUT the 100px padding
   - {OUTPUT_CSV} should have one fewer row (the one you deleted)
""")
 
 
if __name__ == "__main__":
    # Clean up any previous test run
    if BASE.exists():
        shutil.rmtree(BASE)
        print(f"Cleaned up previous test run at {BASE}")
 
    build_test_environment()
    run_step1()
    print_step2_instructions()