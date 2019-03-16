from itertools import islice
import pandas as pd
import numpy as np

from datasets_import import *
from methods import *
from measures import *
from drawing import *

# import warnings
# warnings.filterwarnings('always')  # F-score warning


database_name = 'diabetes'

X, Y = get_dataset(database_name)

folds = [2, 3, 5, 10]
measures = []
measures_for_k = []

for k in folds:
    X_splitted, Y_splitted = split_data_to_chunks(X, Y, k)
    for i in range(k):
        print(f'{i+1}/{k} part of dataset:')
        X_train, Y_train, X_test, Y_test = get_train_and_test_data(X_splitted, Y_splitted, i)

        means_and_stds = bayes_gaussian(X_train, Y_train)
        predicted_classes = get_predicted_classes_gaussian(X_test, Y, means_and_stds)

        measures_local = count_measures(Y_test, predicted_classes)
        measures_for_k.append(list(measures_local))
        # print(f'  Real classes:      {Y_test}')
        # print(f'  Predicted classes: {predicted_classes}\n')
        print_measures(measures_local, '  ')
        print_and_count_measures(Y_test, predicted_classes)

    measures.append(count_measure_avg(np.array(measures_for_k)))

print(measures)
measures = np.array(measures)
draw_by_measures(measures, folds, database_name=database_name)
