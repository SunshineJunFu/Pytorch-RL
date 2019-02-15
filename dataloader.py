#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-11 17:19:12
# @Author  : Jun Fu 
# @Email   : fujun@mail.ustc.edu.cn

import tensorflow as tf 
import numpy as np

class dataloader(object):

	def __init__(self, sess, batch_size):

		super(dataloader, self).__init__()

		self.sess = sess

		self.trainFeatures = np.random.randn(100, 1)

		self.trainLabels = np.random.randn(100, 1)

		self.validFeatures = np.random.randn(100, 1)

		self.validLabels = np.random.randn(100, 1)

		# load dataset #

		self.train_features = tf.placeholder(self.trainFeatures.dtype, self.trainFeatures.shape)

		self.train_labels =  tf.placeholder(self.trainLabels.dtype, self.trainLabels.shape)

		self.train_dataset = tf.data.Dataset.from_tensor_slices((self.train_features, self.train_labels))

		self.train_dataset = self.train_dataset.repeat() # 不断重复

		self.train_iterator = self.train_dataset.make_initializable_iterator()

		self.train_iterator_string = self.sess.run(self.train_iterator.string_handle())

		self.valid_features = tf.placeholder(self.validFeatures.dtype, self.validFeatures.shape)

		self.valid_labels =  tf.placeholder(self.validLabels.dtype, self.validLabels.shape)

		self.valid_dataset = tf.data.Dataset.from_tensor_slices((self.valid_features, self.valid_labels))

		self.valid_iterator = self.valid_dataset.make_initializable_iterator()

		self.valid_iterator_string = self.sess.run(self.valid_iterator.string_handle())

		self.handle = tf.placeholder(tf.string, shape=[])

		self.iterator = tf.data.Iterator.from_string_handle(self.handle,self.train_dataset.output_types, self.train_dataset.output_shapes) # attention

		self.next_item = self.iterator.get_next()

	def get_item(self, train=True):
		
		if train:
			return self.sess.run(self.next_item,feed_dict={self.handle:self.train_iterator_string})
		else:
			return self.sess.run(self.next_item,feed_dict={self.handle:self.valid_iterator_string})	


	def initialize_iterator(self, train=True):

		if train:
			self.sess.run(self.train_iterator.initializer, feed_dict={self.train_features:self.trainFeatures, self.train_labels:self.trainLabels})
		else:
			self.sess.run(self.valid_iterator.initializer, feed_dict={self.valid_features:self.validFeatures,self.valid_labels:self.validLabels})

if __name__ == '__main__':

	with tf.Session() as sess:

		d  = dataloader(sess, 1)

		d.initialize_iterator(0)

		for i in range(200):

			try:

				print(d.get_item(0))

			except tf.errors.OutOfRangeError:

				d.initialize_iterator(0)
		
