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
''' Digitizing methods '''


def get_column_digitized(column, bins):
    atr_digitized = np.digitize(column, bins)
    atr_bincount = np.bincount(atr_digitized)[1:]
    atr_bincount = np.add(atr_bincount, np.ones(shape=(atr_bincount.shape), dtype=np.int))
    return atr_digitized, atr_bincount, bins


def digitize_equally(column, nr_of_bins=10, bins=[]):
    if len(bins) == 0:
        bins = np.linspace(np.min(column), np.max(column), num=nr_of_bins+1, endpoint=True)
    return get_column_digitized(column, bins)


def digitize_by_frequency(column, nr_of_bins=10, bins=[]):
    if len(bins) == 0:
        column = copy.deepcopy(column)
        column.sort()
        bins = np.asarray(list(column)[::int(column.shape[0]/nr_of_bins)])
    return get_column_digitized(column, bins)


def digitize_kmeans(column, nr_of_bins=10, bins=[]):
    if len(bins) == 0:
        discretizer = KBinsDiscretizer(n_bins=nr_of_bins, encode='ordinal', strategy='kmeans')
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


''' Methods related to normal bayes '''


def count_conditional_probabilities(column, Y, c, bins):
    atr_in_class = column[Y == c]
    atr_in_class_bincount = np.bincount(np.digitize(atr_in_class, bins), minlength=len(bins) + 1)
    classes, classes_bincounts = np.unique(Y, return_counts=True)
    atr_in_class_prob = atr_in_class_bincount / list(classes_bincounts)[list(classes).index(c)]
    return atr_in_class_prob


def bayes_digitized(X, Y, attributes_bins):
    classes = np.unique(Y)
    attributes_probs = []

    for i in range(X.shape[1]):
        column = X[:, i]
        probs = []
        for c in classes:
            probs.append(count_conditional_probabilities(column, Y, c, attributes_bins[i]))
        attributes_probs.append(np.array(probs).T)

    return attributes_probs


def get_predicted_classes_normally(X_test, Y, attributes_probs, attributes_bins, digitize):
    classes, classes_bincounts = np.unique(Y, return_counts=True)
    classes_probs = classes_bincounts / len(Y)
    probs_multiplied = np.ones((X_test.shape[0], len(classes)))

    for i in range(X_test.shape[1]):
        column = X_test[:, i]
        atr_digitized, atr_bincount, bins = digitize(column, bins=attributes_bins[i])
        probs_multiplied *= attributes_probs[i][atr_digitized]

    probs_multiplied *= classes_probs
    return classes[probs_multiplied.argmax(axis=1)]


''' Methods related to gaussian bayes '''


def bayes_gaussian(X, Y):
    classes = np.unique(Y)
    means_and_stds = np.empty(shape=(classes.shape[0], 2, X.shape[1]))
    for c in range(classes.shape[0]):
        means_and_stds[c, 0] = np.mean(X[Y == classes[c]], axis=0)
        means_and_stds[c, 1] = np.std(X[Y == classes[c]], axis=0) + 0.001  # smoothing
    return means_and_stds


def get_predicted_classes_gaussian(X_test, Y, means_and_stds):
    classes, classes_bincounts = np.unique(Y, return_counts=True)
    classes_probs = classes_bincounts / len(Y)
    probs_multiplied = np.ones((X_test.shape[0], len(classes)))

    for i in range(X_test.shape[1]):
        column = X_test[:, i]
        means, stds = means_and_stds[:, 0, i].reshape(1, -1), means_and_stds[:, 1, i].reshape(1, -1)

        probs = (1/(stds * np.sqrt(2*np.pi))) * np.exp(- np.power(column.reshape(-1, 1) - means, 2) / np.power(2 * stds, 2))
        probs_multiplied *= probs

    probs_multiplied *= classes_probs
    return classes[probs_multiplied.argmax(axis=1)]


''' Methods related to cross validation '''


def get_train_and_test_data(X, Y, i):
    X_train, X_test = split_set(X, i)
    Y_train, Y_test = split_set(Y, i)
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)


def split_set(set, i):
    test = set[i]
    train = np.concatenate([set[j] for j in range(len(set)) if j != i])
    return train, test


def split_data_to_chunks(X, Y, k):
    data_len = Y.shape[0]
    X_splitted, Y_splitted = [[] for _ in range(k)], [[] for _ in range(k)]
    for i in range(data_len):
        X_splitted[i%k].append(X[i])
        Y_splitted[i%k].append(Y[i])
    # if data_len <= k:
    #     indices = np.arange(1, data_len)
    # else:
    #     indices = np.arange(0, data_len-1, int(data_len / k))[1:]
    # X_splitted = np.vsplit(X, indices)
    # Y_splitted = np.split(Y, indices)
    return X_splitted, Y_splitted


def split_data_stratified(X, Y, k):
    classes = np.unique(Y)
    X_splitted, Y_splitted = [], []
    for c in classes:
        X_of_class_c = X[Y == c]
        X_of_class_c_splitted, Y_of_class_c_splitted = split_data_to_chunks(X_of_class_c, Y[Y == c], k)
        X_splitted.append(X_of_class_c_splitted)
        Y_splitted.append(Y_of_class_c_splitted)

    X_res, Y_res = [[] for _ in range(k)], [[] for _ in range(k)]
    for fold in range(k):
        for c in range(len(classes)):
            X_res[fold] += X_splitted[c][fold]
            Y_res[fold] += Y_splitted[c][fold]
        X_res[fold] = np.array(X_res[fold])

    # for fold in range(k):
    #     X_res.append(np.vstack([X_splitted[i][fold] for i in range(len(classes))]))
    #     Y_res.append(np.concatenate([Y_splitted[i][fold] for i in range(len(classes))]))

    return np.array(X_res), np.array(Y_res)
