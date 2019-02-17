from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

datagen = ImageDataGenerator(
		 featurewise_center=False,  # set input mean to 0 over the dataset
		 samplewise_center=False,  # set each sample mean to 0
		 featurewise_std_normalization=False,  # divide inputs by std of the dataset
		 samplewise_std_normalization=False,  # divide each input by its std
		 zca_whitening=False,  # apply ZCA whitening
		 rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
		 zoom_range = 0.2, # Randomly zoom image
		 width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
		 height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
		 horizontal_flip=True,  # randomly flip images
		 vertical_flip=False, # randomly flip images
		 data_format='channels_last')

img = image.load_img("../train/daisy/0.jpg", target_size=(200, 200))
x = np.array([np.array(img)])
datagen.fit(x)
i = 0
buf = [np.array(img)]
for batch in datagen.flow(x,batch_size=1):
#	plt.imshow(batch[0].astype(np.uint8))
#	plt.show()
	buf.append(batch[0].astype(np.uint8))
	i = i + 1
	if(i > 3):
		break
fig,ax=plt.subplots(1,4)
for i in range(1):
	for j in range (4):
		ax[j].imshow(buf[j])
		if(i == 0 and j == 0):
			ax[j].set_title('raw data')
		else:
			ax[j].set_title('sample ' + str(j))

plt.tight_layout()
plt.show()
