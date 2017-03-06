from __future__ import division
import pickle as pk
import numpy as np 
from coresets import Coresets
from inference import Inference
from two_sample_test import TwoSampleTest
import matplotlib.pyplot as plt




def generate_samples(c_list, coreset, infer):
	sample = dict()
	for key,value in enumerate(c_list):
		print key, value, 'start'
		temp_coreset = coreset.construct_coreset(c = value, epsilon=0.24, delta=0.01)
		random_indices = np.random.choice(coreset.N, size=temp_coreset['coreset_size'], replace=False)
		random_set = {
						'coreset_dimension': coreset.D, 
						'coreset_data': coreset.z_data[random_indices],
						'coreset_weights': None,
						'coreset_size': temp_coreset['coreset_size']
						}
		theta_coreset = infer.mala(coreset=temp_coreset, steps=20, thin=1, warmup=10)
		theta_random = infer.mala(coreset=random_set, steps=20, thin=1, warmup=10)

		sample[key] = {
				'temp_coreset': temp_coreset,
				'random_set': random_set, 
				'theta_coreset': theta_coreset,
				'theta_random': theta_random
				}
	pk.dump(sample, open('sample.data','wb'))

	return 



if __name__ == '__main__':
	# *********** initialze ***************#
	data = np.load('ds1.100_train.npz')
	coreset = Coresets(data,K=6)
	c_list = [0.1/10, 0.1/4, 0.1/3.3, 0.1/1.68, 0.1, 0.1/0.6672 ]
	infer = Inference()
	generate_samples(c_list, coreset, infer)
	sample = pk.load(open('sample.data', 'r'))
	test = TwoSampleTest()
	index = []
	mmd_coreset_list, mmd_random_list = [], []
	loglikelihood_coreset, loglikelihood_random = [],[]

	for key in sample.iterkeys():
		theta_coreset = sample[key]['theta_coreset']
		theta_random = sample[key]['theta_random']
		temp_coreset = sample[key]['temp_coreset']
		random_set = sample[key]['random_set']
		test.set_theta(theta_coreset)
		sample_coreset = test.test_data()
		mmd_coreset = test._estimate_mmd(unbiased=False)
		mmd_coreset_list.append(mmd_coreset)
		test.set_theta(theta_random)
		sample_random = test.test_data()
		mmd_random = test._estimate_mmd(unbiased=False)
		mmd_random_list.append(mmd_random)
		index.append(sample[key]['temp_coreset']['coreset_size'])
		loglikelihood_coreset.append(infer.logistic_likelihood(theta_coreset, sample_coreset['z_test_infer'], 
			weights=None, sum_result=True))
		# print infer.logistic_likelihood(theta_coreset, sample_coreset['z_test_infer'], 
		# 	weights=None, sum_result=False)
		loglikelihood_random.append(infer.logistic_likelihood(theta_random, sample_random['z_test_infer'],
			weights=None, sum_result=True))
	test_size = sample_coreset['z_test_infer'].shape[0]
	loglikelihood_coreset = map(lambda x: -x/test_size, loglikelihood_coreset)
	loglikelihood_random = map(lambda x: -x/test_size, loglikelihood_random)
	index = map(lambda x: np.log10(x), index)
	pk.dump([mmd_coreset_list, mmd_random_list, index, loglikelihood_coreset, loglikelihood_random], open('mmd.data','wb'))
	mmd = pk.load(open('mmd.data', 'r'))

	plt.plot(index, mmd[0], label='coreset', linewidth=3, color = '#FA8072')
	plt.plot(index, mmd[1], label='random', linewidth=3, color='#00CED1')
	plt.xlabel('Size(10^x)',fontsize=15,verticalalignment = 'top', horizontalalignment = 'center',rotation = 0)
	plt.ylabel('MMD', fontsize = 15, verticalalignment = 'bottom', horizontalalignment = 'right',rotation=90)
	plt.legend(loc = 'best')
	# plt.plot(index, mmd[3], label='coreset', linewidth=3, color = '#FA8072')
	# plt.plot(index, mmd[4], label='random', linewidth=3, color='#00CED1')
	# plt.xlabel('Size(10^x)',fontsize=15,verticalalignment = 'top', horizontalalignment = 'center',rotation = 0)
	# plt.ylabel('neg_test_ll', fontsize = 15, verticalalignment = 'bottom', horizontalalignment = 'right',rotation=90)
	# plt.legend(loc = 'best')
	plt.show()
	




