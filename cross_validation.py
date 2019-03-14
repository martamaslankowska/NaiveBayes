from itertools import islice

import pandas as pd
import numpy as np

from datasets_import import *
from methods import *
from measures import *


digitize_methods = [digitize_column_cut]
digitize = digitize_methods[0]
nr_of_bins = 10


X, Y = get_dataset('wine')
Y = digitize_classes(Y)

k = 5
X_splitted, Y_splitted = split_data_to_chunks(X, Y, k)

for i in range(k):
    print(f'\n{i+1}/{k} part of dataset:')
    X_train, Y_train, X_test, Y_test = get_train_and_test_data(X_splitted, Y_splitted, i)
    attributes_probs, attributes_bins = bayes(X, Y, digitize, nr_of_bins)
    predicted_classes = get_X_test_classes(X_test, Y, attributes_probs, attributes_bins, digitize)

    print(f'  Real classes:      {Y_test}')
    print(f'  Predicted classes: {predicted_classes}\n')
    print_measures(Y_test, predicted_classes, '  ')
