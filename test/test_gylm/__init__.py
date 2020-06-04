import os
from benchml.test import TestMock

class TestGylmxx(TestMock):
    group = "gylm"
    data_dir = ("..", "test_data")
    path = os.path.dirname(__file__)

