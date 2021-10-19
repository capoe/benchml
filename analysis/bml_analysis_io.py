"""
for parsing
"""

import gzip
import itertools
import json

import numpy as np


def get_split_props(split_id):
    keys_values = {s[0]: s[1] for s in map(lambda kv: kv.split("="), split_id.split(";"))}
    f = list(map(float, keys_values["train:test"].split(":")))
    keys_values["train_fraction"] = float(f[0] / (f[0] + f[1]))
    keys_values["id"] = split_id
    return keys_values


def parse_single(section, verbose=True):
    dataset = section["dataset"]
    model = section["model"]
    if verbose:
        print("Parse section: dataset=%s, model=%s" % (dataset, model))
    splits = map(get_split_props, section["splits"])
    # print("splits", splits)
    splits_test = list(filter(lambda s: s["perf"] == "test", splits))
    # print("splits_test:", splits_test)
    output_by_train_test_ratio = {}
    for split in splits_test:
        if verbose:
            print(
                "  - We are now parsing the predictions for the split with train_fraction=%1.2f"
                % split["train_fraction"]
            )
        n_repeats = len(section["output"][split["id"]])
        if verbose:
            print(" (repeated %d times)" % n_repeats)
        output_by_train_test_ratio[split["train:test"]] = {
            "train_fraction": split["train_fraction"],
            "n_repeats": n_repeats,
        }
        # print(" Collate test indices, test predictions, test targets")
        test_idcs = np.array(
            list(itertools.chain(*[slice for slice in section["slices"][split["id"]]]))
        ).reshape((-1, 1))
        y_test_pred = np.array(
            list(itertools.chain(*[out["pred"] for out in section["output"][split["id"]]]))
        ).reshape((-1, 1))
        y_test_true = np.array(
            list(itertools.chain(*[out["true"] for out in section["output"][split["id"]]]))
        ).reshape((-1, 1))
        # print(np.shape(test_idcs))
        collated = np.concatenate(
            [
                test_idcs,
                y_test_pred,
                y_test_true,
                np.repeat(np.arange(n_repeats), len(test_idcs) / n_repeats).reshape((-1, 1)),
                np.zeros(len(test_idcs)).reshape((-1, 1)),
            ],
            axis=1,
        )
        # print("    Added output table of shape %s" %repr(collated.shape))

        output_by_train_test_ratio[split["train:test"]].update({"test": collated})

    # We can do the same for the training splits
    # splits_train = list(filter(lambda s: s["perf"] == "train",splits))
    # for split in splits_train:
    # ...
    splits = map(get_split_props, section["splits"])
    splits_train = list(filter(lambda s: s["perf"] == "train", splits))
    # print("splits_train:", splits_train)
    for split in splits_train:
        if verbose:
            print(
                "  - We are now parsing the predictions for the split with train_fraction=%1.2f"
                % split["train_fraction"]
            )
        n_repeats = len(section["output"][split["id"]])
        if verbose:
            print(" (repeated %d times)" % n_repeats)
        # if verbose: print(" Collate train indices, train predictions, train targets")
        train_idcs = np.array(
            list(itertools.chain(*[slice for slice in section["slices"][split["id"]]]))
        ).reshape((-1, 1))
        y_train_pred = np.array(
            list(itertools.chain(*[out["pred"] for out in section["output"][split["id"]]]))
        ).reshape((-1, 1))
        y_train_true = np.array(
            list(itertools.chain(*[out["true"] for out in section["output"][split["id"]]]))
        ).reshape((-1, 1))
        collated = np.concatenate(
            [
                train_idcs,
                y_train_pred,
                y_train_true,
                np.repeat(np.arange(n_repeats), len(train_idcs) / n_repeats).reshape((-1, 1)),
                np.ones(len(train_idcs)).reshape((-1, 1)),
            ],
            axis=1,
        )
        # if verbose: print("    Added output table of shape %s" %repr(collated.shape))

        output_by_train_test_ratio[split["train:test"]].update({"train": collated})
    if verbose:
        print("######################################################################")
    # if verbose: print("- Return a dictionary where key=train:test ratio,value=prediction table")
    return {"model": model, "dataset": dataset}, output_by_train_test_ratio


def parse(benchfile, verbose=False):
    with gzip.open(benchfile, "rt", encoding="utf-8") as zipfile:
        bench = json.load(zipfile)
    by_model_name = {}
    for section in bench:
        meta, train_test_output = parse_single(section, verbose)
        by_model_name[meta["model"]] = train_test_output
    return by_model_name
