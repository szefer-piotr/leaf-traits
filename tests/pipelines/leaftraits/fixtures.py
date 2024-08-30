import pytest
import pandas as pd

@pytest.fixture(scope="module")
def dummy_train_data():
    return pd.DataFrame({
        "id": [1042297, 195616822],
    })

@pytest.fixture(scope="module")
def test_train_data_load():
    test_dataset = pd.read_csv("/home/piotr/leaf-traits/leaf-traits/data/01_raw/train.csv")
    return test_dataset