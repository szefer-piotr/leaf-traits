"""
This is a boilerplate pipeline 'leaftraits'
generated using Kedro 0.19.6
"""

import fsspec
from pathlib import Path

def leaftraits():
    print("Hello Kedro!")
    return 1

def download_data_from_github():
    data_path = Path("data/01_raw")
    image_path = data_path / "train_images"
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, downloading from GitHub...")
        fs = fsspec.filesystem("github", org="szefer-piotr", repo="ltdata")
        fs.get(fs.ls("train_data"), image_path.as_posix(), recursive=True)
        fs.get("train.csv", data_path.as_posix())
    return 1