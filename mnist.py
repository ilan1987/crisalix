import csv


import cv2
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from lenet import LeNet,functional_Lenet
import numpy as np
import tensorflow as tf


if __name__=='__main__':

	weightsPath = None
	logdir = "logs/scalar"


	print("downloading MNIST...")
	with open('mnist/mnist-in-csv/mnist_train.csv','r') as f_p:
		train_ = list(csv.reader(f_p))
	train_ = np.array(train_[1:]).astype(np.uint8)
	trainData = np.reshape(train_[:,1:],(-1,28,28,1))/255
	trainLabels = train_[:,0]

	with open('mnist/mnist-in-csv/mnist_test.csv','r') as f_p:
		train_ = list(csv.reader(f_p))
	train_ = np.array(train_[1:]).astype(np.uint8)
	testData = np.reshape(train_[:,1:],(-1,28,28,1))/255
	testLabels = train_[:,0]


	# reshape the MNIST dataset from a flat list of 784-dim vectors, to
	# 28 x 28 pixel images, then scale the data to the range [0, 1.0]
	# and construct the training and testing splits


	# transform the training and testing labels into vectors in the
	# range [0, classes] -- this generates a vector for each label,
	# where the index of the label is set to `1` and all other entries
	# to `0`; in the case of MNIST, there are 10 class labels
	trainLabels = np_utils.to_categorical(trainLabels, 10)
	testLabels = np_utils.to_categorical(testLabels, 10)
	tensorboard_callback = TensorBoard(log_dir=logdir)

	# initialize the optimizer and model
	print("[INFO] compiling model...")
	lr = 0.1
	opt = SGD(lr=lr)
	model = functional_Lenet.build(width=28, height=28, depth=1, classes=10)
	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])

	# if no weights specified train the model
	if weightsPath is None:
		print("[INFO] training...")
		model.fit(trainData, trainLabels, batch_size=128, nb_epoch=15,
			verbose=1,validation_data=(testData,testLabels),callbacks=[tensorboard_callback])

		# show the accuracy on the testing set
		print("[INFO] evaluating...")
		(loss, accuracy) = model.evaluate(testData, testLabels,
			batch_size=128, verbose=1)
		print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

		print("[INFO] dumping weights to file...")
		weightsPath = "weights/lenet_weights.hdf5"

		model.save_weights(weightsPath, overwrite=True)

	# randomly select a few testing digits
	for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
		# classify the digit
		probs = model.predict(testData[np.newaxis, i])
		prediction = probs.argmax(axis=1)

		# resize the image from a 28 x 28 to 96 x 96
		image = (testData[i][0] * 255).astype("uint8")
		image = cv2.merge([image] * 3)
		image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
		cv2.putText(image, str(prediction[0]), (5, 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

		# show the image and prediction
		print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
			np.argmax(testLabels[i])))
		cv2.imshow("Digit", image)
		cv2.waitKey(0)
