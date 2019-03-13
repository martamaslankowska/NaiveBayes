import numpy as np
import pandas as pd


def import_dataset(name, separator=','):
    dataset = pd.read_csv("datasets/" + name + ".csv", sep=separator, header=None)
    dataset = dataset.sample(frac=1)  # shuffle
    return dataset


def get_attributes_and_classes(dataset, class_column, columns_to_cut=None):
    Y = dataset.loc[:, class_column].values
    X = dataset.loc[:, dataset.columns != class_column].values
    if columns_to_cut:
        X = np.hstack((X[:, :columns_to_cut], X[:, columns_to_cut+1:]))
    return X, Y


def get_attributes_and_classes_from_csv(name, class_column, columns_to_cut=None, separator=','):
    dataset = import_dataset(name, separator)
    X, Y = get_attributes_and_classes(dataset, class_column, columns_to_cut)
    return X, Y


if __name__ == "__main__":
    wine = pd.read_csv("datasets/wine.csv", header=0)
    print(wine.head())
