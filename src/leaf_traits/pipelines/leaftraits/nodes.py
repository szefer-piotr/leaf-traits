"""
This is a boilerplate pipeline 'leaftraits'
generated using Kedro 0.19.6
"""

import fsspec
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Callable

def leaftraits():
    print("Hello Kedro!")
    return 1

def download_data_from_github():
    
    data_path = Path("data/01_raw")
    image_path = data_path / "train_images"
    train_path = data_path / "train.csv"
    
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    
    else:
        print(f"Did not find {image_path} directory, downloading from GitHub...")
        fs = fsspec.filesystem("github", org="szefer-piotr", repo="ltdata")
        fs.get(fs.ls("train_images"), image_path.as_posix(), recursive=True)
        fs.get("train.csv", train_path.as_posix())
    
    return 1

def list_files(
        partitioned_file_list: Dict[str,Callable[[], Any]], limit: int = -1
) -> pd.DataFrame:
    results = []

    for partition_key, partition_load_func in sorted(partitioned_file_list.items()):
        file_path = partition_load_func()
        results.append(file_path)
    
    df = pd.DataFrame(results)

    return df if limit < 0 else df.sample(n=limit, random_state=42)