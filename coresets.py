from __future__ import division
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import k_means
import pickle as pk


class Coresets(object):

	def __init__(self, data, K=6):
		self.data = data
		self.K = K 
		self.x_data = self.data['X']
		self.y_data = self.data['y']
		self.z_data = np.zeros((self.x_data.shape[0],self.x_data.shape[1]))
		self.N = self.x_data.shape[0] #data size
		self.D = self.x_data.shape[1] #feature size

	def _calculate_cluster_information(self):
		for i in range(self.N):
			self.z_data[i] = self.x_data[i] * self.y_data[i] 
		cluster_centers, assignments, inertia = k_means(self.z_data, self.K)
		cluster_size = []
		temp = {key:[] for key in range(self.K)}
		for i in range(self.N):
			temp[assignments[i]].append(self.z_data[i])
		for i in range(self.K):
			cluster_size.append(len(temp[i]))
		cluster_information = {
				'assignments':assignments, 
				'cluster_size':cluster_size,
				'cluster_means':cluster_centers
				}	
		return cluster_information	

	def _calculate_sensitivies(self, R=None):
		cluster_information = self._calculate_cluster_information()
		assignments = cluster_information['assignments']
		cluster_size = cluster_information['cluster_size']
		cluster_means = cluster_information['cluster_means']
		if not R:
			distance = 0
			for i in range(self.N):
				k = assignments[i]
				distance += euclidean_distances([self.z_data[i]], [cluster_means[k]], squared=True)
			I = distance / self.N
			R = 3.0 / np.sqrt(I)
		log_cluster_sizes = np.log(cluster_size)
		dists = np.zeros(self.N)
		denominators = np.ones(self.N)
		sensitivities = np.zeros(self.N)
		for i in range(self.N):
			k = assignments[i]
			true_mean = cluster_means[k,:].copy()
			cluster_means[k,:] *= cluster_size[k] / (cluster_size[k] - 1.0)
			cluster_means[k,:] -= self.z_data[i] / (cluster_size[k] - 1.0)
			dists = euclidean_distances([self.z_data[i]],cluster_means)
			exp_arg = log_cluster_sizes - R * dists
			denominators[i] += np.sum(np.exp(exp_arg))
			cluster_means[k,:] = true_mean
			sensitivities[i] = self.N / denominators[i]
		return sensitivities

	def construct_coreset(self, c, epsilon, delta, R=None):

		sensitivities = self._calculate_sensitivies(R)
		mean_sensitivity = np.sum(sensitivities) / self.N
		coreset_size = np.ceil(c * mean_sensitivity / epsilon**2 * ((self.D + 1) * np.log10(mean_sensitivity) - np.log10(delta)))
		probabilities = sensitivities / np.sum(sensitivities)
		coreset_indices = np.random.choice(self.N, size=coreset_size, p=probabilities, replace=False)
		coreset_weights = 1.0 / probabilities[coreset_indices] / coreset_size
		coreset_weights = coreset_weights 
		# coreset_weights = 1.0 / probabilities[coreset_indices]
		# coreset_weights = coreset_weights / np.sum(coreset_weights)
		coreset = {
					'coreset_dimension': self.D, 
					'coreset_data': self.z_data[coreset_indices,:].copy(),
					'coreset_weights': coreset_weights,
					'coreset_size': coreset_size
					}
		return coreset








if __name__ == '__main__':
	
	# *********** generate the original coreset ****************#
	# data = np.load('ds1.100_train.npz')
	# coresets = Coresets(data, K=6)
	# coreset = coresets.construct_coreset(c = 0.1/4, epsilon=0.24, delta=0.01)
	# pk.dump(coreset, open('coreset.data', 'wb'))

	#*********** generate the original coreset ****************#
	coreset = pk.load(open('coreset.data', 'r'))
	print coreset['coreset_size']
	theta = pk.load(open('theta.data', 'r'))
	infer = Inference()
	test = TwoSampleTest(theta)
	mmd = test._estimate_mmd(unbiased=False)
	print mmd
	# mmd = test.two_sample_test(sample1, sample2, num_shuffles=1000)
	# theta = infer.mala(coreset=coreset, steps=20, thin=1, warmup=10)
	# pk.dump(theta,open('theta.data','wb'))


