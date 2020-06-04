import os
from benchml.test import TestMock

class TestDscribe(TestMock):
    group = "dscribe"
    data_dir = ("..", "test_data")
    path = os.path.dirname(__file__)

