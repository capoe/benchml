import numpy as np

import benchml as bml
from benchml.data import Dataset, ExttDataset, load_dataset


def info_method_tester(d):
    """Test info() method."""
    d_info = d.info()
    assert isinstance(d_info, str)
    assert len(d_info) > 0
    return d_info


def test_extt_load_dataset():
    d = load_dataset("tests/data/ising.extt")

    # Test slicing
    assert isinstance(d[0], np.ndarray)
    assert all(d[0] == d.arrays["X"][0])

    assert isinstance(d[[0, 1]], ExttDataset)

    for key in d.meta.keys():
        assert d[key] == d.meta[key]

    info_example = (
        "ExttDataset with 4 arrays: Array[X(101, 100)] "
        "Array[Y(101,)] Array[J(100, 100)] Array[V(100,)]"
    )
    assert info_example == info_method_tester(d)


def test_xyz_load_dataset():
    d = load_dataset("test/test_data/molecular/set_1.xyz")

    assert isinstance(d, Dataset)

    info_example = (
        "UNNAMED                         #configs=420    task=UNKNOWN   metrics=   std=nan"
    )
    assert info_example == info_method_tester(d)


def test_Dataset():
    d = Dataset()
    # Test "info" method
    info_example = (
        "UNNAMED                         #configs=0      task=UNKNOWN   metrics=   std=nan"
    )
    assert info_example == info_method_tester(d)
    assert len(d) == 0
    assert len(list(d)) == 0


def test_ExttDataset():
    d = ExttDataset()
    # Test "info" method
    info_example = "ExttDataset with 0 arrays:"
    assert info_example == info_method_tester(d)
    assert len(d) == 0
    assert len(list(d)) == 0


def test_ase():
    bml.readwrite.configure(use_ase=True)
    d = load_dataset("test/test_data/molecular/set_1.xyz", index=":")
    assert len(d) == 420
    assert len(list(d)) == 420
