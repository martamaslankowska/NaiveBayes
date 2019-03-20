from datasets_import import *
from measures import *
from drawing import *
from parameters import *

# database names: (0) IRIS, (1) WINE, (2) GLASS and (4) DIABETES
database_name = database_names[3]
X, Y = get_dataset(database_name)

measures_all = []
measures_for_k = []

if bayes_type == bayes_types[0]:  # probabilities (normal way)
    Y = digitize_classes(Y)
    attributes_bins = digitize_X(X, digitize, nr_of_bins)

for k in folds:
    if cross_val_type == cross_val_types[0]:
        X_splitted, Y_splitted = split_data_to_chunks(X, Y, k)
    else:
        X_splitted, Y_splitted = split_data_stratified(X, Y, k)
    for i in range(k):
        print(f'{i+1}/{k} part of dataset:')
        X_train, Y_train, X_test, Y_test = get_train_and_test_data(X_splitted, Y_splitted, i)

        if bayes_type == bayes_types[0]:  # probabilities (normal way)
            attributes_probs = bayes_digitized(X_train, Y_train, attributes_bins)
            predicted_classes = get_predicted_classes_normally(X_test, Y, attributes_probs, attributes_bins, digitize)

        if bayes_type == bayes_types[1]:  # normal distribution
            means_and_stds = bayes_gaussian(X_train, Y_train)
            predicted_classes = get_predicted_classes_gaussian(X_test, Y, means_and_stds)

        measures_local = count_measures(Y_test, predicted_classes)
        measures_for_k.append(list(measures_local))
        print_and_count_measures(Y_test, predicted_classes)

    measures_all.append(count_measure_avg(np.array(measures_for_k)))

print_analysis_info(database_name)
measures = np.array(measures_all)
draw_by_measures(measures, folds, database_name=database_name)
