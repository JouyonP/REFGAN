

from __future__ import division, print_function, absolute_import

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import os, time, itertools, imageio, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data





import numpy

import tensorflow as tf
import tensorflow.contrib.slim as slim

import mnist_data
import cnn_model


MODEL_DIRECTORY = "model/model.ckpt"
LOGS_DIRECTORY = "logs/train"

# Params for Train

training_epochs = 10# 10 for augmented training data, 20 for training data
TRAIN_BATCH_SIZE = 100
display_step = 100
validation_step = 500

# Params for test
TEST_BATCH_SIZE = 1000


# Some parameters
batch_size = TRAIN_BATCH_SIZE
num_labels = mnist_data.NUM_LABELS

# Prepare mnist data
train_total_data, train_size, validation_data, validation_labels, test_data, test_labels = mnist_data.prepare_MNIST_data(True)
train_size = 1

# Boolean for MODE of train or test
is_training = tf.placeholder(tf.bool, name='MODE')

# tf Graph input
x_c = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10]) #answer

# Predict
y = cnn_model.CNN(x_c)

# Get loss of model
with tf.name_scope("LOSS"):
	loss = slim.losses.softmax_cross_entropy(y,y_)

# Create a summary to monitor loss tensor
tf.summary.scalar('loss', loss)

# Define optimizer
with tf.name_scope("ADAM"):
	# Optimizer: set up a variable that's incremented once per batch and
	# controls the learning rate decay.
	batch = tf.Variable(0)

	learning_rate = tf.train.exponential_decay(
		1e-4,  # Base learning rate.
		batch * batch_size,  # Current index into the dataset.
		train_size,  # Decay step.
		0.95,  # Decay rate.
		staircase=True)
	# Use simple momentum for the optimization.
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=batch)

# Create a summary to monitor learning_rate tensor
tf.summary.scalar('learning_rate', learning_rate)

# Get accuracy of model
with tf.name_scope("ACC"):
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create a summary to monitor accuracy tensor
tf.summary.scalar('acc', accuracy)

# Merge all summaries into a single op
#merged_summary_op = tf.merge_all_summaries()

# Add ops to save and restore all the variables
#saver = tf.train.Saver()
#sess = tf.InteractiveSession()
#sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})






def lrelu(x, th=0.2):
	return tf.maximum(th * x, x)

# G(z)
def generator(x, isTrain=True, reuse=False):
	with tf.variable_scope('generator', reuse=reuse):

		# 1st hidden layer
		conv1 = tf.layers.conv2d_transpose(x, 1024, [4, 4], strides=(1, 1), padding='valid')
		lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

		# 2nd hidden layer
		conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [4, 4], strides=(2, 2), padding='same')
		lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

		# 3rd hidden layer
		conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [4, 4], strides=(2, 2), padding='same')
		lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

		# 4th hidden layer
		conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding='same')
		lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

		# output layer
		conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding='same')
		o = tf.nn.tanh(conv5)

		return o

# D(x)
def discriminator(x, isTrain=True, reuse=False):
	with tf.variable_scope('discriminator', reuse=reuse):
		# 1st hidden layer
		conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same')
		lrelu1 = lrelu(conv1, 0.2)

		# 2nd hidden layer
		conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
		lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

		# 3rd hidden layer
		conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')
		lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

		# 4th hidden layer
		conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(2, 2), padding='same')
		lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

		# output layer
		conv5 = tf.layers.conv2d(lrelu4, 1, [4, 4], strides=(1, 1), padding='valid')
		o = tf.nn.sigmoid(conv5)

		return o, conv5

fixed_z_ = np.random.normal(0, 1, (25, 1, 1, 100))
def show_result(num_epoch, show = False, save = False, path = 'result.png'):
	test_images = sess.run(G_z, {z: fixed_z_, isTrain: False})

	size_figure_grid = 5
	fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
	for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
		ax[i, j].get_xaxis().set_visible(False)
		ax[i, j].get_yaxis().set_visible(False)

	for k in range(size_figure_grid*size_figure_grid):
		i = k // size_figure_grid
		j = k % size_figure_grid
		ax[i, j].cla()
		ax[i, j].imshow(np.reshape(test_images[k], (64, 64)), cmap='gray')

	label = 'Epoch {0}'.format(num_epoch)
	fig.text(0.5, 0.04, label, ha='center')

	if save:
		plt.savefig(path)

	if show:
		plt.show()
	else:
		plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
	x = range(len(hist['D_losses']))

	y1 = hist['D_losses']
	y2 = hist['G_losses']

	plt.plot(x, y1, label='D_loss')
	plt.plot(x, y2, label='G_loss')

	plt.xlabel('Epoch')
	plt.ylabel('Loss')

	plt.legend(loc=4)
	plt.grid(True)
	plt.tight_layout()

	if save:
		plt.savefig(path)

	if show:
		plt.show()
	else:
		plt.close()


# load MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])

# variables : input
x = tf.placeholder(tf.float32, shape=(None, 64, 64, 1))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
isTrain = tf.placeholder(dtype=tf.bool)

#Saver


# networks : generator
G_z = generator(z, isTrain)

# networks : discriminator
D_real, D_real_logits = discriminator(x, isTrain)
D_fake, D_fake_logits = discriminator(G_z, isTrain, reuse=True)

# classification
logit_init = tf.placeholder(tf.int32)
logits = cnn_model.CNN(tf.image.resize_images(G_z, [28, 28]), is_training=False)
grads = tf.gradients(logits[0][logit_init] - tf.reduce_sum(logits[0] * (tf.ones([10]) - tf.one_hot([logit_init], 10))), z)
#grads = tf.gradients(logits, z)


# loss for each network
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1]))) # + 

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

lr = 0.0002
# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
	D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
	G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
#sess = tf.InteractiveSession()
#sess = tf.Session()
#saver = tf.train.Saver()
#sess.run(tf.global_variables_initializer())
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})
saver = tf.train.Saver()



#################ckpt_path = saver.restore(sess, tf.train.latest_checkpoint("check"))
#tf.global_variables_initializer().run()

# MNIST resize and normalization
train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval()
train_set = (train_set - 0.5) / 0.5  # normalization; range: -1 ~ 1

# results save folder
root = 'MNIST_DCGAN_results/'
model = 'MNIST_DCGAN_'
if not os.path.isdir(root):
	os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
	os.mkdir(root + 'Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []
# Training cycle
total_batch = int(train_size / batch_size)

# op to write logs to Tensorboard
#summary_writer = tf.train.SummaryWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())

# Save the maximum accuracy value for validation data
max_acc = 0.
'''
for epoch in range(training_epochs):

	# Random shuffling
	numpy.random.shuffle(train_total_data)
	train_data_ = train_total_data[:, :-num_labels]
	train_labels_ = train_total_data[:, -num_labels:]

	# Loop over all batches
	for i in range(total_batch):

		# Compute the offset of the current minibatch in the data.
		offset = (i * batch_size) % (train_size)
		batch_xs = train_data_[offset:(offset + batch_size), :]
		batch_ys = train_labels_[offset:(offset + batch_size), :]

		# Run optimization op (backprop), loss op (to get loss value)
		# and summary nodes
		_, train_accuracy = sess.run([train_step, accuracy] , feed_dict={x_c: batch_xs, y_: batch_ys, is_training: True})

		# Write logs at every iteration
		#summary_writer.add_summary(summary, epoch * total_batch + i)

		# Display logs
		if i % display_step == 0:
			print("Epoch:", '%04d,' % (epoch + 1),
			"batch_index %4d/%4d, training accuracy %.5f" % (i, total_batch, train_accuracy))

		# Get accuracy for validation data
		if i % validation_step == 0:
			# Calculate accuracy
			validation_accuracy = sess.run(accuracy,
			feed_dict={x_c: validation_data, y_: validation_labels, is_training: False})

			print("Epoch:", '%04d,' % (epoch + 1),
			"batch_index %4d/%4d, validation accuracy %.5f" % (i, total_batch, validation_accuracy))

		# Save the current model if the maximum accuracy is updated
		if validation_accuracy > max_acc:
			max_acc = validation_accuracy
'''
# Restore variables from disk
#saver.restore(sess, MODEL_DIRECTORY)

# Calculate accuracy for all mnist test images

test_size = test_labels.shape[0]
batch_size = TEST_BATCH_SIZE
total_batch = int(test_size / batch_size)

acc_buffer = []

# Loop over all batches

for i in range(total_batch):
	# Compute the offset of the current minibatch in the data.
	offset = (i * batch_size) % (test_size)
	batch_xs = test_data[offset:(offset + batch_size), :]
	batch_ys = test_labels[offset:(offset + batch_size), :]

	y_final = sess.run(y, feed_dict={x_c: batch_xs, y_: batch_ys, is_training: False})
	correct_prediction = numpy.equal(numpy.argmax(y_final, 1), numpy.argmax(batch_ys, 1))
	acc_buffer.append(numpy.sum(correct_prediction) / batch_size)

print("test accuracy for the stored model: %g" % numpy.mean(acc_buffer))





# training-loop
np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()

#load
#saver = tf.train.import_meta_graph('check/model19.meta')
#ckpt_path = saver.restore(sess, 'check/model-19')
#ckpt_path = saver.restore(sess, tf.train.latest_checkpoint("check"))


# training parameters
batch_size = 100
train_epoch = 20
'''
for epoch in range(train_epoch):
	G_losses = []
	D_losses = []
	epoch_start_time = time.time()
	for iter in range(mnist.train.num_examples // batch_size):
		# update discriminator
		x_ = train_set[iter*batch_size:(iter+1)*batch_size]
		z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))

		loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, isTrain: True})
		D_losses.append(loss_d_)

		# update generator
		z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
		loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: x_, isTrain: True})
		G_losses.append(loss_g_)

	epoch_end_time = time.time()
	per_epoch_ptime = epoch_end_time - epoch_start_time
	print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
	fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
	show_result((epoch + 1), save=True, path=fixed_p)
	train_hist['D_losses'].append(np.mean(D_losses))
	train_hist['G_losses'].append(np.mean(G_losses))
	train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

	ckpt_path = saver.save(sess, "check/model", epoch)
'''
end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

#save_path = saver.save(sess, "check/model"+str(epoch)+".ckpt")
ckpt_path = saver.restore(sess, 'check/model-19')


#############################
# Finding optimal z for label
#############################
for nn in xrange(100):
	z_search = np.random.normal(0, 1, (1, 1, 1, 100))
	LR_search = 0.01
	logit_init_value = sess.run(logits, {z: z_search, isTrain: False, logit_init: 1})
	print("Initial Label: " + str(np.argmax(logit_init_value)))
	start_time = time.time()
	print(start_time)
	os.mkdir("test_retrain/"+str(start_time)+"/")

	for n in xrange(10):
		test_image, test_label, test_grads = sess.run([G_z, logits, grads], {z: z_search, isTrain: False, logit_init: np.argmax(logit_init_value)})
		print(str(n) + "th trial\nlabel: " + str(np.argmax(test_label)))
		print(test_label)
		plt.imshow(np.reshape(test_image, [64, 64]))
		#plt.savefig("test/"+str(start_time)+"/init_"+str(np.argmax(logit_init_value))+"_current_"+str(np.argmax(test_label))+"_epoch_"+str(n)+".png")
		plt.savefig("test_retrain/"+str(start_time)+"/init_"+str(np.argmax(logit_init_value))+"_current_"+str(np.argmax(test_label))+"_epoch_"+str(n)+".png")
		#print(test_grads)
		#print(z_search)
		#print(np.shape(test_grads))
		#print(np.shape(z_search))
		print(test_grads)
		z_search += LR_search * np.reshape(test_grads, [1, 1, 1, 100])
'''

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")
with open(root + model + 'train_hist.pkl', 'wb') as f:
	pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

images = []
for e in range(train_epoch):
	img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
	images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)
'''
sess.close()


