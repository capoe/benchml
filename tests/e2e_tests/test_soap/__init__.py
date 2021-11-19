import os

from benchml.test import TestMock


class TestSoap(TestMock):
    model_regex = ["bmol_soap_smart_cross_int_krr"]
    data_dir = ("..", "test_data", "molecular_tiny")
    path = os.path.dirname(__file__)
