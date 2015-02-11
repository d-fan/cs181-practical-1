from fann2 import libfann as fann

connection_rate = 1
learning_rate = 0.7
num_input = 256
num_hidden = 256
num_output = 1

target_error = 0.2
max_training = 100
training_per_report = 5

ann = fann.neural_net()
ann.create_sparse_array(connection_rate, (num_input, num_hidden, num_output))
ann.set_learning_rate(learning_rate)

ann.train_on_file("train_fann.txt", max_training, training_per_report, target_error)

ann.save("train.net")