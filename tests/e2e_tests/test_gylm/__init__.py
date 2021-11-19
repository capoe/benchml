import os

from benchml.test import TestMock


class TestGylm(TestMock):
    model_regex = ["bmol_gylm_standard_int_krr"]
    data_dir = ("..", "test_data", "molecular_tiny")
    path = os.path.dirname(__file__)
