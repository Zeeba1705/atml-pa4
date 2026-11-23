import os

import pytest

from common.io_utils import resolve_data_root


def make_repo(path):
    os.makedirs(os.path.join(path, "data"))


def test_data_root_inside_code_repo(tmp_path):
    #local layout: atml-pa4/atml-pa4-datasets
    code= tmp_path / "atml-pa4"
    make_repo(code / "atml-pa4-datasets")

    assert resolve_data_root("atml-pa4-datasets", root=str(code)) == str(code / "atml-pa4-datasets")


def test_data_root_next_to_code_repo(tmp_path):
    #colab/kaggle layout: /content/atml-pa4 and /content/atml-pa4-datasets
    code= tmp_path / "atml-pa4"
    os.makedirs(code)
    make_repo(tmp_path / "atml-pa4-datasets")

    assert resolve_data_root("atml-pa4-datasets", root=str(code)) == str(tmp_path / "atml-pa4-datasets")


def test_data_root_missing_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_data_root("atml-pa4-datasets", root=str(tmp_path / "atml-pa4"))
