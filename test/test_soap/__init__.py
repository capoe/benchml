import os
from benchml.test import TestMock

class TestSoap(TestMock):
    group = "soap"
    data_dir = ("..", "test_data")
    path = os.path.dirname(__file__)

