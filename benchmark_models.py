import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
tf.get_logger().setLevel("ERROR")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dropout
from keras.optimizers import SGD

from helpers import BenchmarkNN, BenchmarkTM

np.random.seed(260795)

with h5py.File('data/200825-130810.h5', 'r') as data:
    print(data.keys())
    print(data['FRB'].shape)
    
    X = np.empty((10000, 128, 512))
    
    t = tqdm(data['FRB'])
    t.set_description('Reading FRBs')
    X[::2] = [a for a in t]
    
    t = tqdm(data['BAK'])
    t.set_description('Reading backgrounds')
    X[1::2] = [a for a in t]
    
    y = np.empty((10000,))
    y[::2] = np.ones(data['FRB'].shape[0])
    y[1::2] = np.zeros(data['BAK'].shape[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=260795)

test_models = []

knn_clf = KNeighborsClassifier(n_neighbors=7)
knn_clf, train_params = BenchmarkTM.train(knn_clf, X_train, y_train, label='K-Nearest Neighbours Classifier')
test_params, roc_curve, big_o_curve = BenchmarkTM.evaluate(knn_clf, X_test, y_test, label='K-Nearest Neighbours Classifier',
                             sample_range=np.arange(0.1, 1.1, .05))
test_models.append({'knn': {'model_loc': 'models/K-Nearest Neighbours Classifier.pkl',
                            'label': 'K-Nearest Neighbours Classifier',
                            'training_parameters': train_params,
                            'testing_parameters': test_params,
                            'cumulative_gain_plot': roc_curve,
                            'big_o_curve': big_o_curve}})

rf_clf = RandomForestClassifier(n_estimators=25)
rf_clf, train_params = BenchmarkTM.train(rf_clf, X_train,  y_train, label='Random Forest Classifier')
test_params, roc_curve, big_o_curve = BenchmarkTM.evaluate(rf_clf, X_test, y_test, label='Random Forest Classifier',
                             sample_range=np.arange(0.1, 1.1, .05))
test_models.append({'rf': {'model_loc': 'models/Random Forest Classifier.pkl',
                           'label': 'Random Forest Classifier',
                           'training_parameters': train_params,
                           'testing_parameters': test_params,
                           'cumulative_gain_plot': roc_curve,
                           'big_o_curve': big_o_curve}})

dl_clf = Sequential()
dl_clf.add(Dense(12, input_dim=128*512, activation='relu'))
dl_clf.add(Dense(24, activation='relu'))
dl_clf.add(Dense(1, activation='sigmoid'))
dl_clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

dl_clf, train_params = BenchmarkNN.train(dl_clf, X_train, y_train, label='Deep Learning Classifier')
test_params, roc_curve, big_o_curve = BenchmarkNN.evaluate(dl_clf, X_test, y_test, label='Deep Learning Classifier',
                                         sample_range=np.arange(0.1, 1.1, .05))

test_models.append({'knn': {'model_loc': 'models/Deep Learning Classifier',
                            'label': 'Deep Learning Classifier',
                            'training_parameters': train_params,
                            'testing_parameters': test_params,
                            'roc_curve': roc_curve,
                            'big_o_curve': big_o_curve}})

X_train = X_train.reshape(-1, 128, 512, 1)
X_test = X_test.reshape(-1, 128, 512, 1)

conv_clf = Sequential()

conv_clf.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(128, 512, 1)))
conv_clf.add(MaxPooling2D(strides=(2,2)))

conv_clf.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
conv_clf.add(MaxPooling2D(strides=(2,2)))

conv_clf.add(Flatten())
conv_clf.add(Dropout(0.2))

conv_clf.add(Dense(12, activation='relu'))
conv_clf.add(Dense(1, activation='sigmoid'))

opt = SGD(lr=0.05)
conv_clf.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
conv_clf, train_params = BenchmarkNN.train(conv_clf, X_train, y_train, label='CNN Classifier')
test_params, roc_curve, big_o_curve = BenchmarkNN.evaluate(conv_clf, X_test, y_test, label='CNN Classifier',
                                         sample_range=np.arange(0.1, 1.1, .05))
test_models.append({'conv_clf': {'model_loc': 'models/CNN Classifier',
                                 'label': 'CNN Classifier',
                                 'training_parameters': train_params,
                                 'testing_parameters': test_params,
                                 'cumulative_gain_plot': roc_curve,
                                 'big_o_curve': big_o_curve}})

plt.figure()
for params in test_models:
    for key, value in params.items():
        t_times, s_sizes = value['testing_parameters']['testing_times'], 				   value['testing_parameters']['sample_sizes']
        plt.plot(s_sizes, t_times, label=value['label'])

plt.legend(loc="best")
plt.xlabel('Sample size')
plt.ylabel('Testing time.')
plt.savefig('media/big-all.png')

with open('raw_data.pkl', 'wb') as fw:
    fw.write(pickle.dumps(test_models))
