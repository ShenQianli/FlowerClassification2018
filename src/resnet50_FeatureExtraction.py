# coding: utf-8
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import cv2
import os
import random as rn
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

np.random.seed(3)
rn.seed(3)
tf.set_random_seed(3)

data = []
X=[]

flower_type = ['daisy', 'rose', 'tulip', 'dandelion', 'sunflower']

model = ResNet50(weights = "../weight/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", include_top = False, pooling = 'avg')

for flower in flower_type:
	data = []
	dir = "../train/" + flower
	print("Feature Extraction (" + flower + ") begin")
	for img in os.listdir(dir):
		path = os.path.join(dir,img)
		img = image.load_img(path, target_size=(224, 224))
		data.append(np.array(img))
	X = np.array(data)
	X = preprocess_input(X)
	feature = model.predict(X)
	np.save("../feature/" + flower + "_resnet50", feature)
	print("Feature Extraction (" + flower+ ") done")

dir = "../test"
data = []
print("Feature Extraction (test) begin")

for i in range(424):
	path = "../test/" + str(i) + ".jpg"
	img = image.load_img(path, target_size=(224, 224))
	data.append(np.array(img))

test = np.array(data)
test = preprocess_input(test)
feature = model.predict(test)
np.save("../feature/" + "test" + "_resnet50", feature)
print("Feature Extraction (test) done")
