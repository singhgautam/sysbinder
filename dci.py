import scipy

import numpy as np

from tqdm import tqdm
from sklearn import ensemble


def gbt(x, y, train_perc=0.75):
    """
    Compute importance matrix.
        x: [num_codes, num_points]
        y: [num_factors, num_points]
        train_perc: float
    """
    num_factors = y.shape[0]
    num_points = y.shape[1]
    num_codes = x.shape[0]
    importance_matrix = np.zeros(shape=[num_codes, num_factors], dtype=np.float64)

    x_train = x[:, :int(num_points * train_perc)]
    x_test = x[:, int(num_points * train_perc):]
    y_train = y[:, :int(num_points * train_perc)]
    y_test = y[:, int(num_points * train_perc):]

    train_loss = []
    test_loss = []
    for i in tqdm(range(num_factors)):
        model = ensemble.GradientBoostingClassifier()
        model.fit(x_train.T, y_train[i, :])
        importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
        test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))

    return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def score(importance_matrix):
    """
    Compute score per code and per factor.
        importance_matrix: [num_codes, num_factors]
    """
    per_code = 1. - scipy.stats.entropy(importance_matrix.T + 1e-11, base=importance_matrix.shape[1])
    per_factor = 1. - scipy.stats.entropy(importance_matrix + 1e-11, base=importance_matrix.shape[0])

    return per_code, per_factor


def dci(importance_matrix):
    """
    Compute DCI Disentanglement and Completeness
        importance_matrix: [num_codes, num_factors]
    """

    per_code, per_factor = score(importance_matrix)

    rhos = importance_matrix.sum(-1)  # [num_codes]
    rhos = rhos / rhos.sum()  # [num_codes]

    dci_modularity = (per_code * rhos).sum()
    dci_compactness = per_factor.mean()

    return dci_modularity, dci_compactness
