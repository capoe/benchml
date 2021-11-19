import os

from benchml.test import TestMock


class TestAcsf(TestMock):
    model_regex = ["bmol_acsf_smart_int_krr"]
    data_dir = ("..", "test_data", "molecular_tiny")
    path = os.path.dirname(__file__)
