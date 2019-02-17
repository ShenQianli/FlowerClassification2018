# coding: utf-8
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os
import random as rn
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import to_categorical
import json
from argparse import ArgumentParser

from cnn_model import naivecnn_model, vgg16_model, vgg19_model
from resnet_model import resnet50_model, resnet101_model, resnet152_model
from densenet_model import densenet121_model, densenet161_model, densenet169_model
from inception_resnet_v2 import InceptionResNetV2
from xception import Xception
from inception_v4 import inception_v4_model

#default arguments
DEFAULT_SEED = 2333
DEFAULT_SAVE_WEIGHT = 0
DEFAULT_SAVE_HISTORY = 0
DEFAULT_SAVE_SUBMISSION = 1
DEFAULT_VALIDATIOM = 0
DEFAULT_VERSION_INFO = ""

def build_parser():
	parser = ArgumentParser()
	parser.add_argument('--model', dest='model', help='model', metavar='MODEL', required=True)
	parser.add_argument('--seed', type = int, dest='seed', help='seed', metavar='SEED', default=DEFAULT_SEED)
	parser.add_argument('--svw', type = int, dest='save_weight', help='save weight or not', metavar='SAVE_WEIGHT', default=DEFAULT_SAVE_WEIGHT)
	parser.add_argument('--svh', type = int, dest='save_history', help='save history or not', metavar='SAVE_HISTORY', default=DEFAULT_SAVE_HISTORY)
	parser.add_argument('--svs', type = int, dest='save_submission', help='save submission or not', metavar='SAVE_SUBMISSION', default=DEFAULT_SAVE_SUBMISSION)
	parser.add_argument('--validation', type = int, dest='validation', help='validation or not', metavar='VALIDATION', default=DEFAULT_VALIDATIOM)
	parser.add_argument('--versioninfo', dest='version_info', help='version information', metavar='VERSION_INFO', default=DEFAULT_VERSION_INFO)
	return parser

#get arguments
parser = build_parser()
options = parser.parse_args()
np.random.seed(options.seed)
rn.seed(options.seed)
tf.set_random_seed(options.seed)
if(options.model in ['NaiveCNN', 'vgg16', 'vgg19', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet161', 'densenet169']):
	IMG_SIZE = 224
else:
	IMG_SIZE = 299
channel = 3
num_classes = 5
versioninfo = ""
if(options.version_info != ""):
	versioninfo = "_" + options.version_info

#load img
print("loading train data...")
flower_type = ['daisy', 'rose', 'tulip', 'dandelion', 'sunflower']
data = []
label = []
for flower in flower_type:
	dir = "../train/" + flower
	for img in tqdm(os.listdir(dir)):
		path = os.path.join(dir,img)
		img = image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
		data.append(np.array(img))
		label.append(flower)

print("loading test data...")
test = []
for i in tqdm(range(424)):
	path = "../test/" + str(i) + ".jpg"
	img = image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
	test.append(np.array(img))

#preprocess & generate label
print("preprocess & generate label...")
X = np.array(data)
test = np.array(test)
if(options.model in ['NaiveCNN', 'vgg16', 'vgg19']):
	from keras.applications.vgg16 import preprocess_input
	X = preprocess_input(X)
	test = preprocess_input(test)
elif(options.model in ['resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet161', 'densenet169']):
	from keras.applications.resnet50 import preprocess_input
	X = preprocess_input(X)
	test = preprocess_input(test)
else:
	X = X / 255.
	X = X - 0.5
	X = X * 2.
	test = test / 255.
	test = test - 0.5
	test = test * 2

le = LabelEncoder()
Y = le.fit_transform(label)
Y = to_categorical(Y,5)

#split train set if Validation
if(options.validation):
	print("split train set...")
	x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=options.seed)
else:
	x_train = X
	y_train = Y

#model
print("build model...")
if(options.model == 'NaiveCNN'):
	model = naivecnn_model(IMG_SIZE, IMG_SIZE, channel, num_classes)
elif(options.model == 'vgg16'):
	model = vgg16_model(IMG_SIZE, IMG_SIZE, channel, num_classes)
elif(options.model == 'vgg19'):
	model = vgg19_model(IMG_SIZE, IMG_SIZE, channel, num_classes)
elif(options.model == 'resnet50'):
	model = resnet50_model(IMG_SIZE, IMG_SIZE, channel, num_classes)
elif(options.model == 'resnet101'):
	model = resnet101_model(IMG_SIZE, IMG_SIZE, channel, num_classes)
elif(options.model == 'resnet151'):
	model = resnet152_model(IMG_SIZE, IMG_SIZE, channel, num_classes)
elif(options.model == 'densenet121'):
	model = densenet121_model(IMG_SIZE, IMG_SIZE, channel, num_classes = num_classes)
elif(options.model == 'densenet161'):
	model = densenet161_model(IMG_SIZE, IMG_SIZE, channel, num_classes = num_classes)
elif(options.model == 'densenet169'):
	model = densenet169_model(IMG_SIZE, IMG_SIZE, channel, num_classes = num_classes)
elif(options.model == 'InceptionResnet'):
	model = InceptionResNetV2(IMG_SIZE, IMG_SIZE, channel, num_classes = num_classes)
elif(options.model == 'xception'):
	model = Xception(IMG_SIZE, IMG_SIZE, channel, num_classes = num_classes)
elif(options.model == 'inceptionv4'):
	model = inception_v4_model(IMG_SIZE, IMG_SIZE, channel, num_classes = num_classes)

print(model.summary())

#data generator
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(x_train)

#train
batch_size=16
epochs=30
if(options.validation):
	epochs = 50
	History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
								  epochs = epochs, validation_data = (x_test,y_test),
								  verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)

else:
	History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
								  epochs = epochs,verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)

#generate submission
if(options.save_submission):
	print("generate submission...")
	predict = np.argmax(model.predict(test) ,axis=1)
	output = pd.DataFrame(le.inverse_transform(predict))
	output.columns
	output.columns = ['Expected']
	output.index.name = "Id"
	output.to_csv("../submission/submission_" + options.model + versioninfo +".csv")

#save weight
if(options.save_weight):
	print("save weight...")
	model.save_weights("../weight/" + options.model + versioninfo + ".h5")

#save history
if(options.save_history):
	print("save history...")
	with open("../history/" + options.model + versioninfo + ".json", 'w') as f:
		json.dump(History.history, f)
