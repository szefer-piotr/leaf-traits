"""
This is a boilerplate test file for pipeline 'leaftraits'
generated using Kedro 0.19.6.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

from src.leaf_traits.pipelines.leaftraits.nodes import (
    download_data_from_github,
    serialize_images,
    train_validation_split,
    train_selected_model
)

def test_download_data_from_github():
    returned_value = download_data_from_github()
    assert returned_value == 1

def test_serialize_images():
    pass

def test_train_validation_split():
    pass

def test_train_selected_model():
    pass