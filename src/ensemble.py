import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing import image
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tqdm import tqdm

path = []

weight = []


OUTPUT = 0
output_path = ""

PRINT = 0

def ensemble(weight):
	combine = np.array([[0. for i in range(5)] for j in range(424)])
	flower_type = ['daisy', 'rose', 'tulip', 'dandelion', 'sunflower']
	le = LabelEncoder()
	le.fit(flower_type)
	if(len(weight)):
		weight = np.array(weight) / sum(weight)
	for i in range(len(path)):
		if(weight[i] == 0): continue
		tmp = pd.read_csv(path[i])
		tmp = np.array(tmp["Expected"])
		tmp = le.transform(tmp)
		tmp = to_categorical(tmp, 5)
		combine += tmp * weight[i]
	result = np.array(le.inverse_transform(np.argmax(combine,axis=1)))
	std = pd.read_csv("../std/std.csv")
	std = np.array(std['Expected'])
	error = 0
	for i in range(424):
		if(np.max(combine[i]) < 0.7 or result[i] != std[i]):
			if(result[i] != std[i]):
				if(PRINT):
					print("X", end = " ")
				error += 1
			else:
				if(PRINT):
					print("O", end = " ")
			if(PRINT):
					print("%03d"%i + " : " + str(combine[i]) + " - result: " + result[i] + " - std: " + std[i])
	print("acc:" + str(1-error/424))
	output = pd.DataFrame(result)
	output.columns = ['Expected']
	output.index.name = 'Id'
	if(OUTPUT):
		output.to_csv(output_path)
	return (1-error/424)

if __name__ == '__main__':
	ensemble(weight)

