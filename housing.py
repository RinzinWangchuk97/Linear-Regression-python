# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 09:31:19 2017

@author: rinziii
"""


import tensorflow as tf
import numpy as np

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

m, n = housing.data.shape

housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

# part 1 using normal equation
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()
    
# part2 manually computing the gradients
tf.reset_default_graph()    
from sklearn.preprocessing import StandardScaler

Scaler = StandardScaler()
scaled_housing_plus_bias=Scaler.fit_transform(housing_data_plus_bias)

n_epochs = 1000
learning_rate = 0.01

tf.reset_default_graph()
X = tf.constant(scaled_housing_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')

theta = tf.Variable(tf.random_uniform([n+1, 1], minval=-1.0, maxval=1.0), name='theta')

y_pred = tf.matmul(X, theta, name='predictions')

error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
gradients = tf.gradients(mse, [theta])[0]
training_op = tf.assign(theta, theta-learning_rate * gradients)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE=", mse.eval())
        sess.run(training_op)
        
    best_theta = theta.eval()
    save_path = saver.save(sess, "\\housing_california\\modelcheckpoint1")
    












