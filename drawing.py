import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker


def draw_by_measures(data, x_range, database_name='', x_label=''):
    data = data*100
    plt.plot(x_range, data[:, 0], 'r+-', label='Accuracy')
    plt.plot(x_range, data[:, 1], 'bs-', label='Precision')
    plt.plot(x_range, data[:, 2], 'g^-', label='Recall')
    plt.plot(x_range, data[:, 3], 'ko-', label='F-score')

    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter())
    plt.xticks(x_range)
    plt.legend()
    plt.title(database_name.capitalize() + ' database\n' if len(database_name) > 0 else '')
    plt.show()
