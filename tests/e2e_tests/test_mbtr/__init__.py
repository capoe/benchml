import os

from benchml.test import TestMock


class TestMbtr(TestMock):
    model_regex = ["bmol_mbtr_int_rr"]
    data_dir = ("..", "test_data", "molecular_tiny")
    path = os.path.dirname(__file__)
