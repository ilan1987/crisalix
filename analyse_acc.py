import csv


import numpy as np
from matplotlib import pyplot as plt

if __name__=='__main__':
	with open('accuracies_obtained/train_ac.csv','r') as f_p:
		reader = csv.reader(f_p)
		train_acc = np.array(list(reader)[1:])[:,2].astype(np.float32)


	with open('accuracies_obtained/val_ac.csv','r') as f_p:
		reader = csv.reader(f_p)
		val_acc = np.array(list(reader)[1:])[:,2].astype(np.float32)
	lrs = [0.01,0.1,0.9,0.9,0.1]
	batchnorm=[False,False,False,True,True]
	train_acc = np.reshape(train_acc,(5,15))
	val_acc = np.reshape(val_acc,(5,15))

	plt.subplots_adjust(0.125,0.1,0.9,0.9,0.3,0.6)
	for i in range(5):
		plt.subplot(3,2,i+1)
		plt.plot(train_acc[i])
		plt.plot(val_acc[i])
		plt.title("Lr={}   BN={}".format(lrs[i],batchnorm[i]))
	plt.savefig('accuracies_obtained/accuracies.png', dpi=100)


