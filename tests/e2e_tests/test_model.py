import optparse

import benchml as bml
from benchml.data import load_dataset

DEFAULT_MODEL = "morgan_rfr"
DEFAULT_DATASET = "tests/e2e_tests/test_data/molecular_tiny/set_1.xyz"
DEFAULT_TARGET = "pactivity"


def test_given_model_fit(
    model_name=DEFAULT_MODEL, dataset_path=DEFAULT_DATASET, target=DEFAULT_TARGET
):
    """Test fit a given model."""
    dataset = load_dataset(dataset_path, meta={"target": target})
    model = bml.models.get(model_name)[0]
    stream = model.open(dataset, verbose=True)
    model.fit(stream, verbose=True)
    output = model.map(stream)
    y_predicted = output["y"]
    model.close(stream)
    assert output is not None
    assert y_predicted is not None


def test_given_model_hyperfit(
    model_name=DEFAULT_MODEL, dataset_path=DEFAULT_DATASET, target=DEFAULT_TARGET
):
    """Test hyperfit of a given model."""
    dataset = load_dataset(dataset_path, meta={"target": target})
    model = bml.models.get(model_name)[0]
    stream = model.open(dataset, verbose=True)
    hyperfit_args = dict(
        split_args=dict(method="random", n_splits=5, train_fraction=0.75),
        accu_args=dict(metric="mae"),
        target="y",
        target_ref="input.y",
        verbose=True,
    )
    model.hyperfit(stream, **hyperfit_args)
    output = model.map(stream)
    y_predicted = output["y"]
    model.close(stream)
    assert output is not None
    assert y_predicted is not None


if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option(
        "-m",
        "--model-name",
        dest="model_name",
        default=DEFAULT_MODEL,
        help="Model name (tag) to test its fit and hyperfit.",
    )
    args, _ = parser.parse_args()
    test_given_model_fit(args.model_name)
    test_given_model_hyperfit(args.model_name)
