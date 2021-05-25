import numpy as np
from time import time
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve, auc


def plot_roc_curve(y_test, y_pred, label=None):
    """
    source: Chengwei
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_score = auc(fpr, tpr)
    roc_plot = plt.figure()
    plt.plot(fpr, tpr, label=f'AUC: {auc_score:.2f}')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(f'media/ROC curve for {label}.png')

    return roc_plot


def train(model, X_train, y_train, label=None):
    """Train the given model and return training accuracy."""
    print(f'Training {label}.')

    if label == 'Deep Learning Classifier':
        start_t = time()
        X_train = np.array([a.flatten() for a in X_train])
        model.fit(X_train, y_train, epochs=5)
        end_t = time() - start_t
    else:
        start_t = time()
        model.fit(X_train, y_train, epochs=10, batch_size=100)
        end_t = time() - start_t

    y_pred = model.predict_classes(X_train)[:, 0]
    training_accuracy = accuracy_score(y_train, y_pred)

    output = f'Training parameters for {label}: \n' \
             f'\t Training time: {end_t:.2f} seconds. \n' \
             f'\t Training Accuracy: {training_accuracy * 100:.2f}% \n' \
             f'{"-" * 20} \n'
    print(output)
    with open(f'models/training_parameters.txt', 'a') as fw:
        fw.writelines(output)

    model.save(f'models/{label}')

    return model, {'training_time': end_t, 'training_accuracy': training_accuracy}


def evaluate(model, X_test, y_test, label=None,
             sample_range=(1., )):
    """Evaluate the model and return performance metrics"""

    if label == 'Deep Learning Classifier':
        start = time()
        X_test = np.array([a.flatten() for a in X_test])
        y_pred = model.predict_classes(X_test)[:, 0]
        end = time() - start
    else:
        start = time()
        y_pred = model.predict_classes(X_test)[:, 0]
        end = time() - start

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    con_mat = confusion_matrix(y_test, y_pred)

    roc_curve_ = plot_roc_curve(y_test, y_pred, label)

    size = X_test.shape[0]
    testing_times = []
    sample_sizes = []

    samples = tqdm(sample_range)
    samples.set_description(f'Benchmarking {label}')
    for i in samples:
        sample_size = int(i * size)
        test_X = X_test[:sample_size]
        start_t = time()
        pred_y = model.predict(test_X)
        end_t = time() - start_t

        testing_times.append(end_t)
        sample_sizes.append(sample_size)

    big_o = plt.figure()
    plt.plot(sample_sizes, testing_times)
    plt.savefig(f'media/Big-O for {label}.png')

    test_params = {'testing_time': end,
                   'accuracy': accuracy,
                   'f1_score': f1,
                   'precision': precision,
                   'recall': recall,
                   'confusion_matrix': con_mat,
                   'testing_times': testing_times,
                   'sample_sizes': sample_sizes}

    output = f'Testing parameters for {label}: \n' \
             f'\t Testing time on {size} samples: {end:.2f} seconds. \n' \
             f'\t Accuracy: {accuracy * 100:.2f}% \n' \
             f'\t Confusion Matrix: \n' \
             f'\t\t {con_mat} \n' \
             f'\t Precision: {precision:.2f} \n' \
             f'\t Recall: {recall:.2f} \n' \
             f'\t F1 Score: {f1:.2f} \n' \
             f'{"-" * 20} \n'
    print(output)
    with open('models/testing_parameters.txt', 'a') as fw:
        fw.write(output)

    return test_params, roc_curve_, big_o


"""
Bibliography

Chengwei, Simple guide on how to generate ROC plot for Keras classifier, 
available at: https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/,
accessed: 13-08-2020
"""
