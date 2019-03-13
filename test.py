import numpy as np
import pandas as pd

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    data = pd.read_csv("datasets/wine.csv")
    # data = data.sample(frac=1)
    print(data.head())

    Y = data["class"].values
    X = data.iloc[:, 1:].values

    column = X[:, 0]
    max_val = np.max(column)
    min_val = np.min(column)
    print(min_val, max_val)

    nr_of_bins = 10
    bound = 0.05  # percent of each bound on the left (min_val) and right (max_val)
    val_diff = max_val - min_val

    bins = np.linspace(min_val - (bound*val_diff), max_val + (bound*val_diff), nr_of_bins)
    # print(bins)
    res = np.digitize(column, bins)
    print(res)

    bincount = np.bincount(res)
    print(bincount)
