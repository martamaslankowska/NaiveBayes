from datasets_import import *
from measures import *
from drawing import *
from parameters import *

# database names: (0) IRIS, (1) WINE, (2) GLASS and (3) DIABETES
database_name = database_names[0]
X, Y = get_dataset(database_name)


for cross_val_type in cross_val_types:
    for b in [5, 10, 15, 20, 25, 40]:
        nr_of_bins = b
        for digitize in digitize_methods:  # probabilities (normal way)
            bayes_type = bayes_types[0]
            measures_all = []
            measures_for_k = []

            Y = digitize_classes(Y)
            attributes_bins = digitize_X(X, digitize, nr_of_bins)

            for k in folds:
                if cross_val_type == cross_val_types[0]:
                    X_splitted, Y_splitted = split_data_to_chunks(X, Y, k)
                else:
                    X_splitted, Y_splitted = split_data_stratified(X, Y, k)
                for i in range(k):
                    # print(f'{i+1}/{k} part of dataset:')
                    X_train, Y_train, X_test, Y_test = get_train_and_test_data(X_splitted, Y_splitted, i)
                    attributes_probs = bayes_digitized(X_train, Y_train, attributes_bins)
                    predicted_classes = get_predicted_classes_normally(X_test, Y, attributes_probs, attributes_bins, digitize)

                    measures_local = count_measures(Y_test, predicted_classes)
                    measures_for_k.append(list(measures_local))
                    # print_and_count_measures(Y_test, predicted_classes)
                measures_all.append(count_measure_avg(np.array(measures_for_k)))
            print_analysis_info(database_name, params=[bayes_type, digitize, nr_of_bins, cross_val_type])
            measures = np.array(measures_all)
            draw_by_measures(measures, folds, database_name=database_name, params=[bayes_type, digitize, nr_of_bins, cross_val_type])#, saving=False)

    bayes_type = bayes_types[1]
    measures_all = []
    measures_for_k = []
    for k in folds:  # gaussian distribution
        if cross_val_type == cross_val_types[0]:
            X_splitted, Y_splitted = split_data_to_chunks(X, Y, k)
        else:
            X_splitted, Y_splitted = split_data_stratified(X, Y, k)
        for i in range(k):
            # print(f'{i + 1}/{k} part of dataset:')
            X_train, Y_train, X_test, Y_test = get_train_and_test_data(X_splitted, Y_splitted, i)
            means_and_stds = bayes_gaussian(X_train, Y_train)
            predicted_classes = get_predicted_classes_gaussian(X_test, Y, means_and_stds)

            measures_local = count_measures(Y_test, predicted_classes)
            measures_for_k.append(list(measures_local))
            # print_and_count_measures(Y_test, predicted_classes)
        measures_all.append(count_measure_avg(np.array(measures_for_k)))
    print_analysis_info(database_name, params=[bayes_type, digitize, nr_of_bins, cross_val_type])
    measures = np.array(measures_all)
    draw_by_measures(measures, folds, database_name=database_name, params=[bayes_type, digitize, nr_of_bins, cross_val_type])#, saving=False)
