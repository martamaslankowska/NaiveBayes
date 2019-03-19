import numpy as np
import pylab as P
import matplotlib.pyplot as plt
from matplotlib import ticker

from datasets_import import get_dataset
from parameters import *


def draw_by_measures(data, x_range, database_name='', params=[], saving=True):
    data = data*100
    plt.plot(x_range, data[:, 1], 'bs-', alpha=0.6, label='Precision')
    plt.plot(x_range, data[:, 2], 'g^-', alpha=0.6, label='Recall')
    plt.plot(x_range, data[:, 3], 'ko-', alpha=0.6, label='F-score')
    plt.plot(x_range, data[:, 0], 'r+-', alpha=0.6, label='Accuracy')

    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter())
    plt.xticks(x_range)
    plt.legend()
    if len(params) > 0:
        plt.title(database_name.upper() + f' DATABASE\n' +
                  (f'{params[1].__name__} with {params[2]} bins'
                   if params[0] == bayes_types[0] else f'{params[0]}') +
                  f' & {params[3]} cross validation'
                  if len(database_name) > 0 else '')
        if saving:
            plt.savefig(f'{database_name}_{params[3]}-cross-val_' +
                        (f'{params[1].__name__}_{params[2]}-bins'
                        if params[0] == bayes_types[0] else f'{params[0]}') + '.png')

    else:
        plt.title(database_name.upper() + f' DATABASE\n' +
                  (f'{digitize.__name__} with {nr_of_bins} bins'
                   if bayes_type == bayes_types[0] else f'{bayes_type}') +
                  f' & {cross_val_type} cross validation'
                  if len(database_name) > 0 else '')
    plt.show()


def draw_by_measures_from_file(x_range, file_name, params, x_range_names=[], x_label='', database_name='', saving=True):
    data = np.load(file_name + '.npy')
    if params[1] < 0:
        data = data[params[0], :, params[2]]
    else:
        data = data[params[0], params[1]]

    x_range = np.array(x_range)
    data = data*100
    plt.bar(x_range-1, data[:, 0], width=0.5, color='r', label='Accuracy')
    plt.bar(x_range-0.33, data[:, 1], width=0.5, color='b', label='Precision')
    plt.bar(x_range+0.33, data[:, 2], width=0.5, color='g', label='Recall')
    plt.bar(x_range+1, data[:, 3], width=0.5, color='k', label='F-score')

    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter())

    y_min, y_max = int(max(np.min(data) - 10, 0)), (np.max(data) + 10) if np.max(data) < 85 else 100
    plt.xticks(x_range, x_range_names, rotation=0)
    plt.xlabel(x_label)
    plt.ylim(y_min, y_max)
    plt.legend()

    plt.title((database_name.upper() + f' DATABASE\n') if database_name != '' else '')
    if saving:
        plt.savefig(f"{database_name}_from-file_'{file_name}'_with_params_{params[0]}_{params[1]}_{params[2]}.png")

    plt.show()


dig_methods = ['equal bins', 'equal frequency', 'K-means', 'gaussian']
draw_by_measures_from_file([4, 8, 12, 16], 'matrix_diabetes_2cross-val_4digitize_4k_4measures_10bins',
                           [1, 1, -1], database_name='diabetes', x_range_names=folds)


def draw_histogram(data, bins=10):
    P.figure()
    n, bins, patches = P.hist(data, bins, normed=1, histtype='bar',
                              color=['crimson', 'burlywood', 'chartreuse'],
                              label=['Crimson', 'Burlywood', 'Chartreuse'])
    P.legend()
    P.show()

    P.figure()
    n, bins, patches = P.hist(data, 10, normed=1, histtype='bar', stacked=True)
    P.show()


def draw_histograms_pandas(X, Y):
    # Import Data
    # df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")
    data = np.hstack((np.array([Y]).T, X))
    df = pd.DataFrame(data)

    # Prepare data
    x_var = 1
    groupby_var = 0
    df_agg = df.loc[:, [x_var, groupby_var]].groupby(groupby_var)
    vals = [df[x_var].values.tolist() for i, df in df_agg]

    # Draw
    plt.figure(figsize=(16, 9), dpi=80)
    colors = [plt.cm.Spectral(i / float(len(vals) - 1)) for i in range(len(vals))]
    n, bins, patches = plt.hist(vals, bins=20, stacked=True,
                                color=colors[:len(vals)])
    plt.show()


database_name = database_names[1]
X, Y = get_dataset(database_name)
draw_histograms_pandas(X, Y)