from methods import *


def print_analysis_info(database_name, params=[]):
    print(database_name.upper(), 'DATABASE')
    if len(params) > 0:
        print(f'method: ' + f'{params[1].__name__} with {params[2]} bins'
              if params[0] == bayes_types[0] else f'{params[0]}')
        print(f'cross val: {params[3]}\n')
    else:
        print(f'method: ' + f'{digitize.__name__} with {nr_of_bins} bins'
              if bayes_type == bayes_types[0] else f'{bayes_type}')
        print(f'cross val: {cross_val_type}\n')


''' Parameters for normal bayes '''
digitize_methods = [digitize_equally, digitize_by_frequency, digitize_kmeans]
digitize = digitize_methods[2]

nr_of_bins = 10

''' Parameters for cross validation '''
folds = [2, 3, 5, 10]
cross_val_types = ['normal', 'stratified']
cross_val_type = cross_val_types[0]

''' Parameters for analyzes '''
database_names = ['iris', 'wine', 'glass', 'diabetes']

bayes_types = ['normal', 'gaussian-distribution']
bayes_type = bayes_types[0]

