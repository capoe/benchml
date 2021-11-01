import numpy as np

from benchml.data import Dataset, ExttDataset, load_dataset


def test_extt_load_dataset():
    d = load_dataset("tests/data/ising.extt")

    # Test slicing
    assert isinstance(d[0], np.ndarray)
    assert all(d[0] == d.arrays["X"][0])

    assert isinstance(d[[0, 1]], ExttDataset)

    for key in d.meta.keys():
        assert d[key] == d.meta[key]


def test_xyz_load_dataset():
    d = load_dataset("test/test_data/molecular/set_1.xyz")

    assert isinstance(d, Dataset)
