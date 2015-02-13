#!/usr/bin/python
"""
Requirements:
>>> sudo apt-get install pip scipy
>>> pip install pybrain
"""

# Pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

# Custom
from common import *

# Standard
from math import sqrt
import pickle

N_NEURON_IN		= 256
N_NEURON_HID	= 64
N_NEURON_OUT	= 1
INPUT_COUNT		= 10000
TRAIN_ITERS		= 1000

def network_rmse(network, data):
	"""
	Returns RMSE of network given data
	"""
	squared_error = 0
	for m in data:
		# Predicts by activating network
		predicted = network.activate(m.features)
		squared_error += (m.gap - predicted)**2
	return sqrt(squared_error / INPUT_COUNT)

# Initialize network
net = buildNetwork(N_NEURON_IN, 
				   N_NEURON_HID, 
				   N_NEURON_OUT)

# Load
molecules = load_molecules(max_count = INPUT_COUNT)

# Initialize and fill dataset
ds = SupervisedDataSet(N_NEURON_IN, N_NEURON_OUT)
for m in molecules:
	ds.addSample(m.features, m.gap)

# Initialize trainer
trainer = BackpropTrainer(net, ds)

# Train TRAIN_ITERS times
print "Training"
for i in xrange(TRAIN_ITERS):
	print "%d: %f" % (i, sqrt(trainer.train()))
	#print "RMSE: %f" % network_rmse(net, molecules)

# Trained at this point. Save to file
print "Saving..."
fileObj = open("%s.pybrain" % INPUT_COUNT, 'w')
pickle.dump(net, fileObj)
fileObj.close()

print "Done"