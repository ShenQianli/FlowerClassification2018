import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt

model_list = ['xception', 'resnet50', 'resnet152', 'densenet121', 'densenet161', 'InceptionResnet']

for model in model_list:
	file = open("../history/" + model + "_validation.json",'r',encoding='utf-8')
	history = json.load(file)
	print(model + "  -acc:" + str(max(history['acc'])) + "  -val_acc:" + str(max(history['val_acc'])))
	plt.plot(history['acc'][:50])
	plt.plot(history['val_acc'][:50])
	plt.legend([model+'_train', model+'_val'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.show()
