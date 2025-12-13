import os
import shutil
import random
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import subprocess

### download data

def download_voc_from_kaggle(dataset="bardiaardakanian/voc0712", dest="VOC_dataset"):
    # Only download if not already present
    if not os.path.exists(dest):
        os.makedirs(dest, exist_ok=True)

        cmd = [
            "kaggle", "datasets", "download",
            "-d", dataset,
            "-p", dest,
            "--unzip"
        ]
        print("Downloading VOC dataset from Kaggle…")
        subprocess.run(cmd, check=True)
        print("Download completed.")
    else:
        print("VOC dataset already exists locally!")

### Format data

# ---------------- CONFIG ----------------
VOC_ROOT = r"VOC_dataset/VOCdevkit"
OUT_ROOT = "dataset"
TRAIN_SPLIT = 0.9
SEED = 42
USE_SETS = ["VOC2007", "VOC2012"]
# ----------------------------------------

random.seed(SEED)

def rf_hash(name):
    return hashlib.md5(name.encode()).hexdigest()[:16]

def convert_xml(xml_path, new_img_name):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # rewrite filename & path
    root.find("filename").text = new_img_name
    path = root.find("path")
    if path is not None:
        path.text = new_img_name

    return tree

def collect_samples():
    samples = []
    for voc in USE_SETS:
        ann_dir = Path(VOC_ROOT) / voc / "Annotations"
        img_dir = Path(VOC_ROOT) / voc / "JPEGImages"

        for xml_file in ann_dir.glob("*.xml"):
            img_file = img_dir / (xml_file.stem + ".jpg")
            if img_file.exists():
                samples.append((img_file, xml_file))
    return samples

def main():
    samples = collect_samples()
    random.shuffle(samples)

    split_idx = int(len(samples) * TRAIN_SPLIT)
    splits = {
        "train": samples[:split_idx],
        "valid": samples[split_idx:]
    }

    for split, items in splits.items():
        out_dir = Path(OUT_ROOT) / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path, xml_path in items:
            base = img_path.stem
            h = rf_hash(base)
            new_name = f"{base}_jpg.rf.{h}"

            # copy image
            shutil.copy(
                img_path,
                out_dir / f"{new_name}.jpg"
            )

            # convert xml
            tree = convert_xml(xml_path, f"{new_name}.jpg")
            tree.write(out_dir / f"{new_name}.xml")

    print("VOC conversion complete")

download_voc_from_kaggle()

main()