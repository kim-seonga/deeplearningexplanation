import tensorflow as tf
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from numpy import array

img = Image.open('2.png')
arr= array(img)
print(arr.shape)
arr = array(img).reshape(28*28, 3)[:, 0:1].reshape(1, 784)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# images = mnist.test.images
# labels = mnist.test.labels

# print(labels[0])

# images = images.reshape(-1,28,28)
# plt.imshow(images[0])
# plt.show()


x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, [None, 10])

w1 = tf.Variable(tf.random_normal([784, 300]))
b1 = tf.Variable(tf.random_normal([300]))

model1 = tf.tanh(tf.matmul(x, w1) + b1)

w = tf.Variable(tf.zeros([300, 10]))
b = tf.Variable(tf.zeros([10]))

model = tf.matmul(model1, w) + b
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(model),1))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
opt = tf.train.AdamOptimizer(0.003).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(300):
        batch_x, batch_y = mnist.train.next_batch(100)
        error, _ = sess.run([cross_entropy, opt], feed_dict={x: batch_x, y: batch_y})
        if i % 100 == 0:
            print("error : %.6f" % error)
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(model, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

    label = sess.run(tf.argmax(y, 1), feed_dict={y: mnist.test.labels})
    y_predict = sess.run(tf.argmax(model, 1), feed_dict={x: mnist.test.images})

    d = {'y': label, 'y_predict': y_predict}
    data = pd.DataFrame(d)
    print(data)

    result = sess.run(tf.argmax(model, 1), feed_dict={x : arr})
    print(result)