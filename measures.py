from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def count_measures(Y_test, Y_pred):
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='macro')
    recall = recall_score(Y_test, Y_pred, average='macro')
    f_score = f1_score(Y_test, Y_pred, average='macro')
    return accuracy, precision, recall, f_score


def print_measures(Y_test, Y_pred, pad=''):
    acc, prec, rec, fsc = count_measures(Y_test, Y_pred)
    print(f'{pad}Accuracy:   {acc * 100:.1f}%')
    print(f'{pad}Precision:  {prec * 100:.1f}%')
    print(f'{pad}Recall:     {rec * 100:.1f}%')
    print(f'{pad}F-score:    {fsc * 100:.1f}%')
