from fann2 import libfann as fann
import sys

if len(sys.argv) < 3:
	print "Usage: " + sys.argv[0] + " [input] [ann save]"
	sys.exit()

connection_rate = 1
learning_rate = 0.7
num_input = 256
num_hidden = 256
num_output = 1

target_error = 0.2
max_epochs = 10
epoch_per_report = 1

ann = fann.neural_net()
ann.create_sparse_array(connection_rate, (num_input, num_hidden, num_output))
ann.set_learning_rate(learning_rate)

ann.train_on_file(sys.argv[1], max_epochs, epoch_per_report, target_error)

ann.save(sys.argv[2])