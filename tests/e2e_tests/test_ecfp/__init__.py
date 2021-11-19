import os

from benchml.test import TestMock


class TestEcfp(TestMock):
    model_regex = ["bmol_ecfp4_.*"]
    data_dir = ("..", "test_data", "molecular_tiny")
    path = os.path.dirname(__file__)
