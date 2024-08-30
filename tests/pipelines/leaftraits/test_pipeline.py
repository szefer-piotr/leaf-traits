"""
This is a boilerplate test file for pipeline 'leaftraits'
generated using Kedro 0.19.6.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

import pandas as pd

from tests.pipelines.leaftraits.fixtures import dummy_train_data
from tests.pipelines.leaftraits.fixtures import test_train_data_load
from src.leaf_traits.pipelines.leaftraits.nodes import (
    download_data_from_github,
    serialize_images,
    train_validation_split,
    train_selected_model
)

def test_download_data_from_github():
    returned_value = download_data_from_github()
    assert returned_value == 1

def test_input_dataframe():
    pass
    # Input for the node is of a correct type
    # assert type(dummy_train_data) is pd.DataFrame
    # Non-empty dataframe
    # assert dummy_train_data.shape[0] != 0
    # It has the required column "file_path" column

def test_serialize_images_returns_the_desired_column(dummy_train_data: pd.DataFrame):

    dummy_path_data = serialize_images(
        train_raw = dummy_train_data,
        image_path = '/home/piotr/leaf-traits/leaf-traits/data/01_raw/train_images'
    )

    # Input for the node is of a correct type
    assert type(dummy_path_data) is pd.DataFrame
    # Non-empty dataframe
    assert dummy_path_data.shape[0] != 0
    # It produces the 'id' column
    assert 'id' in dummy_path_data.columns


def test_serialize_images_returns_a_dataframe_using_test_df(test_train_data_load: pd.DataFrame):

    assert type(test_train_data_load) is pd.DataFrame
    
    # dummy_path_data = serialize_images(
    #     train_raw = test_train_data,
    #     image_path = '/home/piotr/leaf-traits/leaf-traits/data/01_raw/train_images'
    # )
    

def test_train_validation_split():
    pass

def test_train_selected_model():
    pass