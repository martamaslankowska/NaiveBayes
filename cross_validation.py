from itertools import islice
import pandas as pd
import numpy as np

from datasets_import import *
from measures import *
from methods import *
from drawing import *

# import warnings
# warnings.filterwarnings('always')  # F-score warning


database_name = 'glass'

digitize_methods = [digitize_equally, digitize_by_frequency, digitize_kmeans]
digitize = digitize_methods[2]
nr_of_bins = 10

X, Y = get_dataset(database_name)
Y = digitize_classes(Y)
attributes_bins = digitize_X(X, digitize, nr_of_bins)

# k = 5
folds = [3, 5, 10]  #[2, 3, 5,
measures_all = []
measures_for_k = []

for k in folds:
    X_splitted, Y_splitted = split_data_stratified(X, Y, k)
    # X_splitted, Y_splitted = split_data_to_chunks(X, Y, k)
    for i in range(k):
        print(f'{i+1}/{k} part of dataset:')
        X_train, Y_train, X_test, Y_test = get_train_and_test_data(X_splitted, Y_splitted, i)
        attributes_probs = bayes_digitized(X_train, Y_train, attributes_bins)
        predicted_classes = get_predicted_classes_normally(X_test, Y, attributes_probs, attributes_bins, digitize)

        measures_local = count_measures(Y_test, predicted_classes)
        measures_for_k.append(list(measures_local))
        # print(f'  Real classes:      {Y_test}')
        # print(f'  Predicted classes: {predicted_classes}\n')
        # print_measures(measures, '  ')
        print_and_count_measures(Y_test, predicted_classes)

    measures_all.append(count_measure_avg(np.array(measures_for_k)))

# print(measures_all)
measures_all = np.array(measures_all)
draw_by_measures(measures_all, folds, database_name=database_name)
print_analysis_info(database_name)
