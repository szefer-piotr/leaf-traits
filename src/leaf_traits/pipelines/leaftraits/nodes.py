"""
This is a boilerplate pipeline 'leaftraits'
generated using Kedro 0.19.6
"""

def leaftraits():
    print("Hello Kedro!")
    return 1

def download_data_from_github():
    # Setup paths to data
    data_path = Path("data/01_raw")
    image_path = data_path / "train_images"
    tabular_path = data_path / "train.csv"

    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, downloading from GitHub...")
        fs = fsspec.filesystem("github", org="szefer-piotr", repo="leaf_trait_data")
        fs.get(fs.ls(""), data_path.as_posix(), recursive=False)
    