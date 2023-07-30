from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, SeparableConv2D, Activation, \
    MaxPooling2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from numpy import loadtxt
import tensorflow.keras.backend as K
from sklearn import metrics
from sklearn import svm
from keras.utils import to_categorical, plot_model
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle
from keras.optimizers import Adam
from imutils import paths
import numpy as np
from PIL import Image
import matplotlib as plt
import os
import random


def normalize_gray_images_data(X):
    return X / 255.0


def normalize_rgb_images_data(X):
    # print((X[0][0][0][0] - np.mean(X[0][0][0]))/255.0)
    # print('Iterations and below method gives the same results')
    # print(((X - np.mean(X, axis=3).reshape((len(X),100, 100, 1)))/255.0)[0][0][0][0])
    return (X - np.mean(X, axis=3).reshape((len(X),100, 100, 1)))/255.0


def get_images_from_dataset(images_folder_path):
    labels_images = []
    for i in range(10):
        image_paths = list(paths.list_images(f"{images_folder_path}/Dataset/{i}"))
        label_images = []
        for path in image_paths:
            im = Image.open(path)
            label_images.append(im.copy())
            im.close()
        labels_images.append(label_images)
    return labels_images


def get_images_data(labels_images, convert_to_gray=True):
    X = []
    y = []
    for label_i in range(len(labels_images)):
        label_images = labels_images[label_i]
        for im in label_images:
            im_converted = im.convert('L' if convert_to_gray else 'RGB').resize((100, 100))
            X.append(np.asarray(im_converted))
            y.append(label_i)

    X = np.array(X)
    y = np.array(y)

    return normalize_gray_images_data(X) if convert_to_gray else normalize_rgb_images_data(X), y


def f_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def do_k_cross_validation(model, X, y, k=5, epochs=50, verbose=1):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    train_accuracies = []
    test_accuracies = []
    for fold_i, (train_indices, test_indices) in enumerate(kf.split(X=X, y=y)):
        print(f'Fold #{fold_i + 1}:')
        x_train, y_train = X[train_indices], y[train_indices]
        x_test, y_test = X[test_indices], y[test_indices]

        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, verbose=verbose)

        _, train_acc, __ = model.evaluate(x_train, y_train, verbose=1)
        _, test_acc, f_score_val = model.evaluate(x_test, y_test, verbose=1)
        print('Train: %.3f, Test: %.3f, F-Score: %.3f\n' % (train_acc, test_acc, f_score_val))

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    print('All folds were trained')
    avg_train_accuracy = np.array(train_accuracies).mean()
    avg_test_accuracy = np.array(test_accuracies).mean()
    print(f'Average Train Accuracy: {avg_train_accuracy}')
    print(f'Average Test Accuracy: {avg_test_accuracy}\n\n')

    return avg_train_accuracy, avg_test_accuracy


# =============================================================
# Read Images

images_folder = os.getcwd() + '/Sign-Language-Digits-Dataset'
labels_images = get_images_from_dataset(images_folder)
X_gray, y_gray = get_images_data(labels_images, convert_to_gray=True)
X_rgb, y_rgb = get_images_data(labels_images, convert_to_gray=False)

# =============================================================
# ANN

X, y = shuffle(X_gray, y_gray, random_state=42)
X = X.reshape(len(X), 100 * 100)
y = to_categorical(y)

modelANN1 = Sequential([
    Dense(80, input_shape=(100 * 100,), activation='relu'),
    Dense(40, activation='relu'),
    Dense(20, activation='relu'),
    Dense(10, activation='softmax')
])
plot_model(modelANN1, show_shapes=True, to_file='ann1-model.png')
modelANN1.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy', f_score])

modelANN2 = Sequential([
    Dense(320, input_shape=(100 * 100,), activation='relu'),
    Dense(160, activation='relu'),
    Dense(80, activation='relu'),
    Dense(40, activation='relu'),
    Dense(20, activation='relu'),
    Dense(10, activation='softmax')
])
plot_model(modelANN2, show_shapes=True, to_file='ann2-model.png')
modelANN2.compile(loss='categorical_crossentropy', optimizer=Adam(),
                  metrics=['accuracy', f_score])

print('ANN Model 1')
do_k_cross_validation(modelANN1, X, y, verbose=1)
print('ANN Model 2')
do_k_cross_validation(modelANN2, X, y, verbose=1)

# =============================================================
# CNN

X, y = shuffle(X_rgb, y_rgb, random_state=42)
y = to_categorical(y)

modelCNN = Sequential([
    Conv2D(128, 3, strides=2, padding='same', input_shape=(100, 100, 3)),
    MaxPooling2D(3, strides=2, padding="same"),
    Activation('relu'),

    Conv2D(256, 3, strides=2, padding='same'),
    MaxPooling2D(3, strides=2, padding="same"),
    Activation('relu'),

    SeparableConv2D(512, 3, padding='same'),
    BatchNormalization(),
    SeparableConv2D(512, 3, padding='same'),
    BatchNormalization(),
    MaxPooling2D(3, strides=2, padding="same"),
    Activation('relu'),

    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
plot_model(modelCNN, show_shapes=True, to_file='cnn-model.png')

modelCNN.compile(loss='categorical_crossentropy', optimizer=Adam(),
                 metrics=['accuracy', f_score])

print('CNN Model')
do_k_cross_validation(modelCNN, X, y, verbose=1, epochs=20)

# =============================================================
# SVM

x_train, x_test, y_train, y_test = train_test_split(
    X_gray, y_gray, random_state=104, test_size=0.2, shuffle=True)
size_x_train = len(x_train)
size_x_test = len(x_test)
x_train = x_train.reshape((size_x_train, 100*100))
x_test = x_test.reshape((size_x_test, 100*100))

model = svm.SVC(kernel='rbf')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print('SVM Scores')
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(
    y_test, y_pred, average="weighted"))
print("Recall:", metrics.recall_score(y_test, y_pred, average="weighted"))
print("Fscore:", metrics.f1_score(y_test, y_pred, average="weighted"))

# =============================================================
# Experiments Reports

# 1- Changing Neural Network Size & Depth -> Model depth (layers number) have more effect on reaching error convergence than it's size (neurons number)
# 2- Trying different loss functions -> Multi-Class (Categorical) Cross Entropy is best for most cases of classification problems with more than 2 classes
# 3- Setting seeds for everything -> The models was still delivering different results due to stochastic nature of ML algorithms
# 4- Used SeparableConv2D and BatchNormalization in CNN model -> They can increase speed of the model a lot but may decrease model accuracy
