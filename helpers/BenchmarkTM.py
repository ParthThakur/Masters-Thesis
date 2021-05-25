import numpy as np
from time import time
from tqdm import tqdm
import pickle

import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import plot_roc_curve


def feature_extract(array):
    """Extract statistical features from given array"""
    max_mean = max(array.mean(axis=1))
    max_std = max(array.std(axis=1))
    max_median = max(np.median(array, axis=1))
    max_skewness = max(skew(array, axis=1))
    max_kurtosis = max(kurtosis(array, axis=1))
    mean_max = array.max(axis=1).mean()
    max_variance = max(array.var(axis=1))
    mean = array.mean()

    return max_mean, max_std, max_median, max_skewness, max_kurtosis, mean_max, max_variance, mean


def train(model, X_train, y_train, label=None):
    """Train the given model and return training accuracy."""
    print(f'Training {label}.')
    start_t = time()
    X_train = np.array([feature_extract(array) for array in X_train])
    model.fit(X_train, y_train)
    end_t = time() - start_t
    train_pred = model.predict(X_train)
    training_accuracy = accuracy_score(y_train, train_pred)

    output = f'Training parameters for {label}: \n' \
             f'\t Training time: {end_t:.2f} seconds. \n' \
             f'\t Training Accuracy: {training_accuracy * 100:.2f}% \n' \
             f'{"-" * 20} \n'
    print(output)
    with open(f'models/training_parameters.txt', 'a') as fw:
        fw.writelines(output)

    with open(f'models/{label}.pkl', 'wb') as fw:
        fw.write(pickle.dumps(model))

    return model, {'training_time': end_t, 'training_accuracy': training_accuracy}


def evaluate(model, X_test, y_test, label=None,
             sample_range=(1., )):
    """Evaluate the model and return performance metrics"""
    start = time()
    X_test_ = np.array([feature_extract(array) for array in X_test])
    y_pred = model.predict(X_test_)
    end = time() - start

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    con_mat = confusion_matrix(y_test, y_pred)

    roc_curve = plot_roc_curve(model, X_test_, y_test)
    plt.title(f'ROC curve for {label}')
    plt.savefig(f'media/ROC curve for {label}.png')

    size = X_test.shape[0]
    testing_times = []
    sample_sizes = []

    samples = tqdm(sample_range)
    samples.set_description(f'Benchmarking {label}')
    for i in samples:
        sample_size = int(i * size)
        test_X = X_test[:sample_size]

        start_t = time()
        feature_set = np.array([feature_extract(array) for array in test_X])
        pred_y = model.predict(feature_set)
        end_t = time() - start_t

        testing_times.append(end_t)
        sample_sizes.append(sample_size)

    big_o = plt.figure()
    plt.plot(sample_sizes, testing_times)
    plt.savefig(f'media/Big-O for {label}.png')
    # plt.show()

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

    return test_params, roc_curve, big_o
