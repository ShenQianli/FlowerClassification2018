from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

flower_type = ['daisy', 'rose', 'tulip', 'dandelion', 'sunflower']

fig,ax=plt.subplots(1,5)

for i in range(5):
	img = image.load_img("../train/" + flower_type[i] + "/2.jpg", target_size=(200, 200))
	ax[i].imshow(img)
	ax[i].set_title(flower_type[i])

plt.tight_layout()
plt.show()
