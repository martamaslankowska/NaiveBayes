import pandas as pd
import numpy as np

from datasets_import import *
from methods import *


digitize_methods = [digitize_column_cut]
digitize = digitize_methods[0]

X, Y = get_dataset('wine')
X_test, Y_test = X[:20], Y[:20]
nr_of_bins = 10

attributes_probs, attributes_bins = bayes(X, Y, digitize, nr_of_bins)
predicted_classes = get_X_test_classes(X_test, Y, attributes_probs, attributes_bins, digitize)

print(f'Real classes:      {Y_test}')
print(f'Predicted classes: {predicted_classes}')

