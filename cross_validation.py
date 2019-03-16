from itertools import islice
import pandas as pd
import numpy as np

from datasets_import import *
from methods import *
from measures import *
from drawing import *

# import warnings
# warnings.filterwarnings('always')  # F-score warning


database_name = 'wine'

digitize_methods = [digitize_column_equally, digitize_column_by_frequency, digitize_column_kmeans]
digitize = digitize_methods[2]
nr_of_bins = 10

measures_for_all_k = []

X, Y = get_dataset(database_name)
Y = digitize_classes(Y)
attributes_bins = digitize_X(X, digitize, nr_of_bins)

# k = 5
folds = [2, 3, 5, 10]
measures = []

for k in folds:
    X_splitted, Y_splitted = split_data_to_chunks(X, Y, k)
    for i in range(k):
        print(f'{i+1}/{k} part of dataset:')
        X_train, Y_train, X_test, Y_test = get_train_and_test_data(X_splitted, Y_splitted, i)
        attributes_probs = bayes(X_train, Y_train, attributes_bins)
        predicted_classes = get_X_test_classes(X_test, Y, attributes_probs, attributes_bins, digitize)

        measures_local = count_measures(Y_test, predicted_classes)
        measures_for_all_k.append(list(measures_local))
        # print(f'  Real classes:      {Y_test}')
        # print(f'  Predicted classes: {predicted_classes}\n')
        # print_measures(measures, '  ')
        print_and_count_measures(Y_test, predicted_classes)

    measures.append(count_measure_avg(np.array(measures_for_all_k)))

print(measures)
measures = np.array(measures)
draw_by_measures(measures, folds, database_name=database_name)
