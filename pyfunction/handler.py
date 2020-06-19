
import random
import matplotlib.pylab as plt

import tensorflow as tf
import numpy as np
import PIL.Image as Image
import json

#!pip install -U tf-hub-nightly
#!pip install tfds-nightly
import tensorflow_hub as hub

from tensorflow.keras import layers

def handle(req):
	classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}

	if req.find("http") == -1:
		print("Give me a URL of a picture and I'll recognize it for you.")
		return

	IMAGE_SHAPE = (224, 224)

	classifier = tf.keras.Sequential([
		hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
	])

	"""### Run it on a single image

	Download a single image to try the model on.
	"""



	grace_hopper = tf.keras.utils.get_file(random.randrange(1000)+'ram.jpg', req)
	grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
	grace_hopper

	grace_hopper = np.array(grace_hopper)/255.0
	grace_hopper.shape

	"""Add a batch dimension, and pass the image to the model."""

	result = classifier.predict(grace_hopper[np.newaxis, ...])
	result.shape

	"""The result is a 1001 element vector of logits, rating the probability of each class for the image.

	So the top class ID can be found with argmax:
	"""

	predicted_class = np.argmax(result[0], axis=-1)
	predicted_class

	"""### Decode the predictions

	We have the predicted class ID,
	Fetch the `ImageNet` labels, and decode the predictions
	"""

	labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
	imagenet_labels = np.array(open(labels_path).read().splitlines())

	plt.imshow(grace_hopper)
	plt.axis('off')
	predicted_class_name = imagenet_labels[predicted_class]
	_ = plt.title("Prediction: " + predicted_class_name.title())
	results = {"prediction:", predicted_class_name.title()}
	print(json.dumps(result))

	return
