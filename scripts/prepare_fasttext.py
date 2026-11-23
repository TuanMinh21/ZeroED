#!/usr/bin/env python3
"""
Downloads and reduces fastText model dimensions to match table column count.
Usage:
    python scripts/prepare_fasttext.py [dataset_name]
Example:
    python scripts/prepare_fasttext.py rayyan
"""
import glob
import os
import shutil
import subprocess
import sys
import time

FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "rayyan"
    clean_csv = f"data/{dataset}_clean.csv"
    if not os.path.isfile(clean_csv):
        print(f"[Error] Dataset file not found: {clean_csv}", file=sys.stderr)
        sys.exit(1)

    import pandas as pd
    df = pd.read_csv(clean_csv, dtype=str)
    n_cols = df.shape[1]
    print(f"Dataset '{dataset}' has {n_cols} columns.")

    cache_dir = os.environ.get("FT_CACHE_DIR", "ft_cache")
    os.makedirs(cache_dir, exist_ok=True)
    reduced_path = os.path.join(cache_dir, f"cc.en.{n_cols}.bin")

    # Check if pre-reduced model already exists
    if os.path.isfile(reduced_path):
        print(f"[FastText] Reduced model found: {reduced_path}")
        print(f"FASTTEXT_MODEL={os.path.abspath(reduced_path)}")
        return

    # Check for pre-reduced models in Kaggle input
    kaggle_reduced = glob.glob(f"/kaggle/input/**/cc.en.{n_cols}.bin", recursive=True)
    if kaggle_reduced:
        print(f"[FastText] Found reduced model in Kaggle input: {kaggle_reduced[0]}")
        print(f"FASTTEXT_MODEL={os.path.abspath(kaggle_reduced[0])}")
        return

    # Locate or download full cc.en.300.bin
    full_path = "cc.en.300.bin"
    if not os.path.isfile(full_path):
        kaggle_full = glob.glob("/kaggle/input/**/cc.en.300.bin", recursive=True)
        if kaggle_full:
            full_path = kaggle_full[0]

    if not os.path.isfile(full_path):
        tmp_dir = "/tmp/fasttext_full"
        os.makedirs(tmp_dir, exist_ok=True)
        full_path = os.path.join(tmp_dir, "cc.en.300.bin")
        gz_path = full_path + ".gz"
        if not os.path.isfile(full_path):
            print(f"[FastText] Downloading cc.en.300.bin.gz (4.2 GB)...")
            subprocess.run(["wget", "-c", "--progress=dot:giga", "-O", gz_path, FASTTEXT_URL], check=True)
            print("[FastText] Decompressing gzip archive...")
            subprocess.run(["gunzip", "-f", gz_path], check=True)

    print(f"[FastText] Loading {full_path} and reducing to {n_cols} dimensions...")
    t0 = time.time()
    import fasttext
    import fasttext.util

    ft = fasttext.load_model(full_path)
    fasttext.util.reduce_model(ft, n_cols)
    ft.save_model(reduced_path)
    del ft
    print(f"[FastText] Saved reduced model: {reduced_path} ({os.path.getsize(reduced_path) / 1e6:.1f} MB) in {time.time()-t0:.1f}s")

    # Clean up temporary 7GB file if downloaded to /tmp
    if full_path.startswith("/tmp/fasttext_full") and os.path.isfile(full_path):
        os.remove(full_path)
        print("[FastText] Removed temporary full model to free disk space.")

    print(f"FASTTEXT_MODEL={os.path.abspath(reduced_path)}")


if __name__ == "__main__":
    main()
