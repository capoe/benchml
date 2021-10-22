"""For processing data."""

import numpy as np
from asaplib.fit import LC_SCOREBOARD
from asaplib.fit.getscore import get_score
from sklearn import linear_model


def model_elementwise_error(model_now, select="test", compare="MSE", replica=None):
    # some options to select certain data points

    if select == "all":
        y_train = model_now["train"]
        y_test = model_now["test"]
        y_raw = np.concatenate([y_test, y_train])
        y_all = y_raw[np.argsort(y_raw[:, 0])]
    elif select == "train":
        if replica is None:
            y_all = model_now["train"]
        else:
            y_all = model_now["train"][model_now["train"][:, -2] == replica]
    elif select == "test":
        if replica is None:
            y_all = model_now["test"]
        else:
            y_all = model_now["test"][model_now["test"][:, -2] == replica]
    else:
        raise ValueError("selection not exist")

    # y_MSE, y_MAE
    if compare == "MSE":
        # train_idcs, y_pred, y_true, submodel_id, train_or_not
        return np.asarray([v[1] - v[2] for v in y_all])
    elif compare == "MAE":
        return np.asarray([np.abs(v[1] - v[2]) for v in y_all])
    elif compare == "RMSE":
        return np.asarray([(v[1] - v[2]) ** 2.0 for v in y_all])
    elif compare == "y":
        return np.asarray([v[1] for v in y_all])
    else:
        raise ValueError("selection not exist")


def model_error_correlation(
    model_1, model_2, select="test", compare="MSE", metric="CORR", replica=None
):

    y_1 = model_elementwise_error(model_1, select, compare, replica)
    y_2 = model_elementwise_error(model_2, select, compare, replica)

    return get_score(y_1, y_2)[metric]


def model_correlation_matrix(
    by_model, key_now, select="test", compare="MSE", metric="CORR", replica=None, verbose=True
):

    correlation_matrix = np.ones((len(by_model.keys()), len(by_model.keys())))
    model_list = []

    for i, model_key_now in enumerate(by_model.keys()):
        model_list.append(model_key_now)

        for j, model_key_now_2 in enumerate(by_model.keys()):
            if i > j:
                model_now = by_model[model_key_now][key_now]
                model_now_2 = by_model[model_key_now_2][key_now]

                correlation_matrix[i, j] = correlation_matrix[j, i] = model_error_correlation(
                    model_now, model_now_2, select, compare, metric, replica
                )
                if verbose:
                    print(model_key_now, model_key_now_2, correlation_matrix[i, j])

    return correlation_matrix, model_list


def all_model_correlation_matrix(
    by_model,
    key_now_list=[],
    select="all",
    compare="MSE",
    metric="PearsonR",
    replica=None,
    verbose=True,
):

    all_corr_matrix_size = len(by_model.keys()) * len(key_now_list)

    all_corr_matrix = np.ones((all_corr_matrix_size, all_corr_matrix_size))
    all_model_list = []

    for i, model_key_now in enumerate(by_model.keys()):
        for ri, r_now in enumerate(key_now_list):
            # the index of the first model
            index_model_1 = i * len(key_now_list) + ri
            all_model_list.append(str(model_key_now) + str(r_now))

            for j, model_key_now_2 in enumerate(by_model.keys()):
                for rj, r_now_2 in enumerate(key_now_list):
                    # the index of the second model
                    index_model_2 = j * len(key_now_list) + rj

                    if i > j:
                        model_now = by_model[model_key_now][r_now]
                        model_now_2 = by_model[model_key_now_2][r_now_2]

                        all_corr_matrix[index_model_1, index_model_2] = all_corr_matrix[
                            index_model_2, index_model_1
                        ] = model_error_correlation(
                            model_now, model_now_2, select, compare, metric, replica
                        )
                    if verbose:
                        print(model_key_now, r_now, model_key_now_2, r_now_2, all_corr_matrix[i, j])

    return all_corr_matrix, all_model_list


def fit_hybrid_lc(by_model, all_model_list, alpha=0.1, sc_name="RMSE", verbose=False):

    lc_scores = LC_SCOREBOARD([])

    clf = linear_model.Ridge(alpha, max_iter=10000, fit_intercept=False)

    for train_frac in by_model[all_model_list[0]].keys():
        n_repeats = by_model[all_model_list[0]][train_frac]["n_repeats"]
        n_train = int(len(by_model[all_model_list[0]][train_frac]["train"]) / n_repeats)
        for replica in range(n_repeats):
            for i, model_now in enumerate(all_model_list):
                # all the original test set
                all_tmp = by_model[model_now][train_frac]["test"]
                all_tmp_0 = all_tmp[all_tmp[:, -2] == replica]
                # split the set into validation and test
                n_validation = int(min(len(all_tmp_0) / 2, 100))
                validation_tmp_0, test_tmp_0 = (
                    all_tmp_0[:n_validation, :],
                    all_tmp_0[n_validation:, :],
                )  # train_test_split(all_tmp_0, test_size=1.-validation_ratio)

                if i == 0:
                    hybrid_X = np.zeros((len(validation_tmp_0[:, 1]), len(all_model_list)))
                    hybrid_y = validation_tmp_0[:, 2]

                    hybrid_X_test = np.zeros((len(test_tmp_0[:, 1]), len(all_model_list)))
                    hybrid_y_test = test_tmp_0[:, 2]

                hybrid_X[:, i] = validation_tmp_0[:, 1]
                hybrid_X_test[:, i] = test_tmp_0[:, 1]

            clf.fit(hybrid_X, hybrid_y)
            if verbose:
                print("# replica ", replica)
            hybrid_y_pred = hybrid_y * 0.0
            hybrid_y_pred_test = hybrid_y_test * 0.0
            for i, model_now in enumerate(all_model_list):
                if clf.coef_[i] > 0.1:
                    if verbose:
                        print(model_now, clf.coef_[i])
                hybrid_y_pred += clf.coef_[i] * hybrid_X[:, i]
                hybrid_y_pred_test += clf.coef_[i] * hybrid_X_test[:, i]
            if verbose:
                print(get_score(hybrid_y, hybrid_y_pred))
                print(get_score(hybrid_y_test, hybrid_y_pred_test))
            lc_scores.add_score(
                n_train + n_validation, get_score(hybrid_y_test, hybrid_y_pred_test)
            )
    return np.asmatrix(lc_scores.fetch(sc_name))
