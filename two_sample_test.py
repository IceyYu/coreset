from __future__ import division
import numpy as np
import pickle as pk
from inference import Inference

class TwoSampleTest(object):

	def __init__(self, theta=None):
		self.theta = theta
		pass

	def set_theta(self, theta):
		self.theta = theta

	def test_data(self):
		test_data = np.load('ds1.100_test.npz')
		x_test_data = test_data['X']
		y_test_data = test_data['y']
		size = x_test_data.shape[0]
		y_test_data.shape = size, 1
		z_test_true = x_test_data * y_test_data
		y_infer = np.zeros(size)
		
		# for i in range(size):
		# 	logistic_likelihood = -np.log1p(np.exp(-(1+x_test_data[i].dot(self.theta)) ** 3))
		# 	if logistic_likelihood > 0:
		# 		y_infer[i] = 1
		# 	else:
		# 		y_infer[i] = -1
		logistic_likelihood = Inference.logistic_likelihood(self.theta, test_data['X'], weights=None, sum_result=False)
		neg_logistic_likelihood = Inference.logistic_likelihood(self.theta, -test_data['X'], weights=None, sum_result=False)

		for i in range(size):
			if logistic_likelihood[i] > neg_logistic_likelihood[i]:
				y_infer[i] = 1
			else:
				y_infer[i] = -1
		y_test_data.shape = size
		num = 0	
		for i in range(len(y_test_data)):
			if y_test_data[i] == y_infer[i]:
				num += 1
		print num
		
		y_infer.shape = size, 1
		z_test_infer = x_test_data * y_infer
		sample = {
					'z_test_true': z_test_true,
					'z_test_infer': z_test_infer
					}
		return sample
		
	def _kernel(self, X, Y=None):
		if Y is None:
			Y = X
		return (1 + X.T.dot(Y))**3
		# return (self.theta + X.T.dot(Y))**3

	def _estimate_mmd(self, unbiased=False):
		sample = self.test_data()
		sample1 = sample['z_test_true']
		sample2 = sample['z_test_infer']

		K11 = self._kernel(sample1, sample1)
		K22 = self._kernel(sample2, sample2)
		K12 = self._kernel(sample1, sample2)
		if unbiased:
			np.fill_diagonal(K11, 0.0)
			np.fill_diagonal(K22, 0.0)
			n = float(np.shape(K11)[0])
			m = float(np.shape(K22)[0])
			return np.sum(K11) / (n**2 - n) + np.sum(K22) / (m**2 - m - 2 * np.mean(K12))
		else:
			return np.mean(K11) + np.mean(K22) - 2 * np.mean(K12)



			