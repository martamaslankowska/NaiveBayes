import copy

import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


# def digitize_column_cut(column, nr_of_bins=10, bins=[]):
#     if len(bins) == 0:
#         _, bins = pd.cut(column, nr_of_bins, retbins=True)
#     atr_digitized = np.digitize(column, bins)
#     atr_bincount = np.bincount(atr_digitized)
#     atr_bincount = np.add(atr_bincount, np.ones(shape=(atr_bincount.shape), dtype=np.int))
#     return atr_digitized, atr_bincount, bins

def get_column_digitized(column, bins):
    atr_digitized = np.digitize(column, bins)
    atr_bincount = np.bincount(atr_digitized)[1:]
    atr_bincount = np.add(atr_bincount, np.ones(shape=(atr_bincount.shape), dtype=np.int))
    return atr_digitized, atr_bincount, bins


def digitize_column_equally(column, nr_of_bins=10, bins=[]):
    if len(bins) == 0:
        bins = np.linspace(np.min(column), np.max(column), num=nr_of_bins+1, endpoint=True)
    return get_column_digitized(column, bins)


def digitize_column_by_frequency(column, nr_of_bins=10, bins=[]):
    if len(bins) == 0:
        column = copy.deepcopy(column)
        column.sort()
        bins = np.asarray(list(column)[::int(column.shape[0]/nr_of_bins)])
        # bins = np.linspace(np.min(column), np.max(column), num=nr_of_bins+1, endpoint=True)
    return get_column_digitized(column, bins)


def digitize_column_kmeans(column, nr_of_bins=10, bins=[]):
    if len(bins) == 0:
        discretizer = KBinsDiscretizer(n_bins=nr_of_bins, encode='ordinal', strategy='uniform')
        discretizer.fit(column.reshape(-1, 1))
        bins = discretizer.bin_edges_[0].astype(float)
    return get_column_digitized(column, bins)


def digitize_X(X, digitize, nr_of_bins):
    attributes_bins = []
    for i in range(X.shape[1]):
        column = X[:, i]
        atr_digitized, atr_bincount, bins = digitize(column, nr_of_bins=nr_of_bins)
        attributes_bins.append(bins)
    return attributes_bins


def digitize_classes(Y):
    _, indexes = np.unique(Y, return_inverse=True)
    return indexes


def count_conditional_probabilities(column, Y, c, bins):
    atr_in_class = column[Y == c]
    atr_in_class_bincount = np.bincount(np.digitize(atr_in_class, bins), minlength=len(bins) + 1)
    classes, classes_bincounts = np.unique(Y, return_counts=True)
    atr_in_class_prob = atr_in_class_bincount / list(classes_bincounts)[list(classes).index(c)]
    return atr_in_class_prob


def bayes(X, Y, attributes_bins):
    classes = np.unique(Y)
    attributes_probs = []

    for i in range(X.shape[1]):
        column = X[:, i]
        probs = []
        for c in classes:
            probs.append(count_conditional_probabilities(column, Y, c, attributes_bins[i]))
        attributes_probs.append(np.array(probs).T)

    return attributes_probs


def get_X_test_classes(X_test, Y, attributes_probs, attributes_bins, digitize):
    classes, classes_bincounts = np.unique(Y, return_counts=True)
    classes_probs = classes_bincounts / len(Y)
    probs_multiplied = np.ones((X_test.shape[0], len(classes)))

    for i in range(X_test.shape[1]):
        column = X_test[:, i]
        atr_digitized, atr_bincount, bins = digitize(column, bins=attributes_bins[i])
        probs_multiplied *= attributes_probs[i][atr_digitized]

    probs_multiplied *= classes_probs
    return classes[probs_multiplied.argmax(axis=1)]


def get_train_and_test_data(X, Y, i):
    X_train, X_test = split_set(X, i)
    Y_train, Y_test = split_set(Y, i)
    return X_train, Y_train, X_test, Y_test


def split_set(set, i):
    test = set[i]
    train = np.concatenate([set[j] for j in range(len(set)) if j != i])
    return train, test


def split_data_to_chunks(X, Y, k):
    data_len = Y.shape[0]
    indices = np.arange(0, data_len, int(data_len / k) + 1)[1:]
    X_splitted = np.vsplit(X, indices)
    Y_splitted = np.split(Y, indices)
    return X_splitted, Y_splitted
