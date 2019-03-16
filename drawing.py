import numpy as np
import matplotlib.pyplot as plt


def draw_by_measures(data, x_range, database_name='', x_label=''):
    print(data[:, 0])
    plt.plot(x_range, data[:, 0], 'r+-', label='Accuracy')
    plt.plot(x_range, data[:, 1], 'bs-', label='Precision')
    plt.plot(x_range, data[:, 2], 'g^-', label='Recall')
    plt.plot(x_range, data[:, 3], 'ko-', label='F-score')
    plt.xticks(x_range)
    plt.legend()
    plt.show()
