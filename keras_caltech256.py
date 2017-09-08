import getopt
import os
import pickle
import re
import sys
import textwrap
from random import shuffle

import cv2
import numpy
from keras import optimizers
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.initializers import RandomUniform
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

helptext = textwrap.dedent("""\
        keras_caltech256.py -p <path>
        
        -p Path to Caltech-256 dataset
        """)

pathToDataset = ''

try:
    opts, args = getopt.getopt(sys.argv[1:], "hp:", ["path="])
except getopt.GetoptError:
    print(helptext)
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print(helptext)
        sys.exit()
    elif opt in ("-p", "--path"):
        pathToDataset = arg

# calculate training data
# ---------------------------------------------------

nb_classes = 257
nb_descriptors = 30

categories = {}
dataset = []
try:
    dataset = pickle.load(open("data/caltech256.dat", "rb"))
    categories = pickle.load(open("data/caltech256_categories.dat", "rb"))
except FileNotFoundError:
    categories = {}
    dataset = []
    orb = cv2.ORB_create(nfeatures = nb_descriptors)
    print("Extracting features...")
    for root, dirs, files in os.walk(pathToDataset, topdown=False):
        for name in files:
            if re.match(r"(.*\.jpg)|(.*\.png)|(.*\.gif)", name):
                categoryName = os.path.basename(root).split('.')[1].replace('-101', '')
                categoryNumber = name.split('_')[0]

                img_gray = cv2.imread(os.path.join(root, name), 0)
                kp = orb.detect(img_gray, None)
                if (len(kp) is not nb_descriptors):
                    print("Not enough keypoints (kp: {})! Image: {}".format(len(kp), name))
                    continue
                kp, des = orb.compute(img_gray, kp)

                categoryVector = numpy.zeros(nb_classes)
                categoryVector[int(categoryNumber) - 1] = 1.0

                if not categoryNumber in categories:
                    categories[categoryNumber] = (categoryVector, categoryName)

                dataset.append([categoryVector, des[:nb_descriptors, ].flatten('C').astype(float)])

    pickle.dump(dataset, open("data/caltech256.dat", "wb"))
    pickle.dump(categories, open("data/caltech256_categories.dat", "wb"))
    print("successfully saved {} image vectors to caltech256.dat".format(len(dataset)))


# init to learn
# ---------------------------------------------------


trainingSize = int(len(dataset) * 0.8)

# remove clutter from dataset
removeClutter = False
if removeClutter:
    clutter = numpy.zeros(nb_classes, dtype=numpy.int)
    clutter[256] = 1
    dataset = list(filter(lambda data: not numpy.array_equal(data[0], clutter), dataset))

shuffle(dataset)

# setting up training and test data
X_train = numpy.array([data[1] for data in dataset[:trainingSize]])
X_train /= 255.0
print("Input dimension: {}".format(len(X_train[0])))
Y_train = numpy.array([data[0] for data in dataset[:trainingSize]])

X_test = numpy.array([data[1] for data in dataset[trainingSize:]])
X_test /= 255.0
Y_test = numpy.array([data[0] for data in dataset[trainingSize:]])


# create neural network
model = Sequential()
model.add(Dense(21, input_dim=len(X_train[0]), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(7, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(21, activation='relu'))
model.add(Dense(len(Y_train[0]), activation="softmax"))

# Compile neural network
model.compile(loss='categorical_crossentropy', optimizer="adadelta", metrics=['accuracy'])

# Train and test neural network
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, batch_size=100, verbose=2)



# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.figure()

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()