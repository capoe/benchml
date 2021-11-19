import pytest

from benchml.readwrite import tokenize_extxyz_meta

header_tokenize_examples = [
    (
        """key1="str1" key2=123 key3="another value" \n""",
        {"key1": "str1", "key2": 123, "key3": "another value"},
    ),
    (
        """SMILES="CCO" MW=44.11 cLogP=2.281 HAC=3 \n""",
        {"SMILES": "CCO", "MW": 44.11, "cLogP": 2.281, "HAC": 3},
    ),
]


@pytest.mark.parametrize("header_example,expected", header_tokenize_examples)
def test_tokenize_extxyz_meta(header_example, expected):
    res = tokenize_extxyz_meta(header_example)
    assert isinstance(res, dict)
    assert res == expected
