# %% LOAD DATA
import pandas as pd
import pickle

training_file = 'train.p'
validation_file= 'valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

sign_names = pd.read_csv("signnames.csv")

# %% GET INFO
n_train = X_train.shape[0]
n_test = X_test.shape[0]
image_shape = X_train[0].shape
n_classes = sign_names.shape[0]
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# %% GRAY SCALE
import random
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def grayscale(data_set):
    gray = np.array([0.299,0.587,0.114]).reshape(3,1)
    return data_set.dot(gray)

X_train = grayscale(X_train)
X_valid = grayscale(X_valid)
X_test = grayscale(X_test)

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

print(sign_names.loc[y_train[index]])
fig, axes = plt.subplots(nrows=2, ncols=1)
ax0, ax1 = axes.flatten()

ax0.hist(image.flatten(), 256, histtype='stepfilled', color='gray', label='red')
ax1.imshow(image, cmap="gray")

fig.tight_layout()
fig.set_size_inches(4.5, 9.5)
plt.show()


# %% HISTOGRAM EQUALIZATION

import numpy as np

def dataset_histogram_equalization(dataset):
    data_equalized = np.zeros(dataset.shape)
    for i in range(dataset.shape[0]):
        image = dataset[i, :, :, 0]
        data_equalized[i, :, :, 0] = image_histogram_equalization(image)[0]
    return data_equalized

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf

X_train = dataset_histogram_equalization(X_train)
X_valid = dataset_histogram_equalization(X_valid)
X_test = dataset_histogram_equalization(X_test)

#sample
image = X_train[index].squeeze()

fig, axes = plt.subplots(nrows=2, ncols=1)
ax0, ax1 = axes.flatten()

ax0.hist(image.flatten(), 256, histtype='stepfilled', color='gray', label='red')
ax1.imshow(image, cmap="gray")

fig.tight_layout()
fig.set_size_inches(4.5, 9.5)
plt.show()

# %% MIN MAX SCALING
import matplotlib.pyplot as plt
import numpy as np

X_train = (X_train - 128) / 128
X_valid = (X_valid - 128) / 128
X_test = (X_test - 128) / 128

fig, ax = plt.subplots()
ax.hist(X_train.reshape(-1), 256, histtype='bar', color='gray', label='min-max')
ax.legend(prop={'size': 10})
ax.set_title('color distribution')

fig.tight_layout()
fig.set_size_inches(9.5, 3.5)
plt.show()


# %% SHUFLE DATA
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)


# %% MODEL ARCHITECTURE
from tensorflow.contrib.layers import flatten

def SignNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    #Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x32.
    #out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
    #out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))
    c1_weights = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 32), mean = mu, stddev = sigma))
    c1_bias = tf.Variable(tf.zeros(32))
    c1 = tf.nn.conv2d(x, c1_weights, strides=[1, 1, 1, 1], padding='VALID') + c1_bias

    #Activation.
    c1 = tf.nn.relu(c1)

    #Pooling. Input = 28x28x32. Output = 14x14x32.
    c1 = tf.nn.max_pool(c1, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #Layer 2: Convolutional. Output = 10x10x16.
    c2_weights = tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 16), mean = mu, stddev = sigma))
    c2_bias = tf.Variable(tf.zeros(16))
    c2 = tf.nn.conv2d(c1, c2_weights, strides=[1, 1, 1, 1], padding='VALID') + c2_bias

    #Activation.
    c2 = tf.nn.relu(c2)

    #Pooling. Input = 10x10x16. Output = 5x5x16.
    c2 = tf.nn.max_pool(c2, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #Flatten. Input = 5x5x16. Output = 400.
    flat = flatten(c2)

    #Layer 3: Fully Connected. Input = 400. Output = 120.
    f1_weights = tf.Variable(tf.truncated_normal(shape=(400,120), mean = mu, stddev = sigma))
    f1_bias = tf.Variable(tf.zeros(120))
    f1 = tf.matmul(flat,f1_weights) + f1_bias

    #Activation.
    f1 = tf.nn.relu(f1)

    #Layer 4: Fully Connected. Input = 120. Output = 84.
    f2_weights = tf.Variable(tf.truncated_normal(shape=(120,84), mean = mu, stddev = sigma))
    f2_bias = tf.Variable(tf.zeros(84))
    f2 = tf.matmul(f1,f2_weights) + f2_bias

    #Activation.
    f2 = tf.nn.relu(f2)

    #Layer 5: Fully Connected. Input = 84. Output = 10.
    out_weights =  tf.Variable(tf.truncated_normal(shape=(84,43), mean = mu, stddev = sigma))
    out_bias = tf.Variable(tf.zeros(43))
    out = tf.matmul(f2,out_weights) + out_bias
    return out

# %% FEATURES AND LABLES
import tensorflow as tf

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

# %% TRAINING PIPELINE
rate = 0.0008

logits = SignNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

# %% MODEL EVALUATION
EPOCHS = 64
BATCH_SIZE = 64

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# %% TRAINING MODEL
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    train_error = []
    valid_error = []

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        train_accuracy = evaluate(X_train, y_train)
        train_error.append(train_accuracy)
        validation_accuracy = evaluate(X_valid, y_valid)
        valid_error.append(validation_accuracy)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './sign-net')
    print("Model saved")

# %% TEST
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
