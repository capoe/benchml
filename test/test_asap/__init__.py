import os
from benchml.test import TestMock

class TestAsap(TestMock):
    group = "asap"
    data_dir = ("..", "test_data")
    path = os.path.dirname(__file__)

