# coding: utf-8
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import random as rn
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import json

np.random.seed(3)
rn.seed(3)
tf.set_random_seed(3)

flower_type = ['daisy', 'rose', 'tulip', 'dandelion', 'sunflower']
label = []
X=[]
Y=[]

for flower in flower_type:
	path = "../feature/" + flower + "_resnet50.npy"
	data = np.load(path)
	for i in range(data.shape[0]):
		X.append(list(data[i]))
		label.append(flower)

le = LabelEncoder()
Y = le.fit_transform(label)
Y = to_categorical(Y,5)
X = np.array(X)

model = Sequential()
model.add(Dense(512, input_shape = (2048, )))
model.add(Activation('relu'))
model.add(Dense(5, activation = "softmax"))

model.compile(optimizer=SGD(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

batch_size=128
epochs=100

History = model.fit(X, Y, epochs = epochs, batch_size=batch_size, verbose = 1, validation_split=0.2)

test = np.load("../feature/test_resnet50.npy")
predict = np.argmax(model.predict(test) ,axis=1)
output = pd.DataFrame(le.inverse_transform(predict))
output.columns = ['Expected']
output.index.name = "Id"

output.to_csv("../submission/submission_resnet50_TF_validation.csv")

with open("../history/resnet50TF_validation.json", 'w') as f:
	json.dump(History.history, f)
