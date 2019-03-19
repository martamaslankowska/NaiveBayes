import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from parameters import *


def draw_by_measures(data, x_range, database_name='', params=[], saving=True):
    data = data*100
    plt.plot(x_range, data[:, 0], 'r+-', label='Accuracy')
    plt.plot(x_range, data[:, 1], 'bs-', label='Precision')
    plt.plot(x_range, data[:, 2], 'g^-', label='Recall')
    plt.plot(x_range, data[:, 3], 'ko-', label='F-score')

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
    # plt.show()
