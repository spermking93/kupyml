# -*- coding: utf-8 -*-



# MNIST 데이터를 다운로드 한다.

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

""" 텐서플로우 예제중 mnist에서 데이터를 입력한다
one_hot 기능 사용한다."""
# TensorFlow 라이브러리를 추가한다.

import tensorflow as tf



# 변수들을 설정한다.

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

"""y=Wx + b 의 회귀 방정식으로 시작한다. 789는 28*28, 10은 1부터 10까지 숫자 갯수"""
# cross-entropy 모델을 설정한다.

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

""" cross_entropy를 사용하여 코스트 함수를 만들고, 
경사하강법 모델을 적용시킨다."""
# 경사하강법으로 모델을 학습한다.
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""sess.run 오류날시 막는 코드"""
init = tf.global_variables_initializer()
"""tf.initialize_all_variables() 은 구버전이라서 에러난다."""

sess = tf.Session()

sess.run(init)

for i in range(1000):

  batch_xs, batch_ys = mnist.train.next_batch(100)

  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

"""변수 초기화후 100개의 무작위 데이터들을 가져와서 train을 돌리는걸 1000번 반복한다."""
# 학습된 모델이 얼마나 정확한지를 출력한다.

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))