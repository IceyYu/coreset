from __future__ import division
import numpy as np 
import scipy.sparse as sp
import numpy.random as npr

class Inference(object):


	def __init__(self):
		pass

	@staticmethod
	def logistic_likelihood(theta, Z, weights=None, sum_result=True):
		if not sp.issparse(Z):
			Z = np.atleast_2d(Z)
		with np.errstate(over='ignore'):  # suppress exp overflow warning
			likelihoods = -np.log1p(np.exp(-Z.dot(theta)))
		if not sum_result:
			return likelihoods
		if weights is not None:
			likelihoods = weights * likelihoods.T/ np.sum(weights)
		return np.sum(likelihoods)

	def _logistic_likelihood_grad(self, theta, Z, weights=None):
		if not sp.issparse(Z):
			Z = np.atleast_2d(Z)
		grad_weights = 1. / (1. + np.exp(-Z.dot(theta)))
		if weights is not None:
			grad_weights *= weights
		if sp.issparse(Z):
			return sp.csr_matrix(grad_weights).dot(Z).toarray().squeeze()
		else:
			return grad_weights.dot(Z)

	def _mh(self, x0, p, q, sample_q, steps=1, warmup=None, thin=1,
       proposal_param=None, target_rate=0.234):
		''' Run (adaptive) MH algorithm '''
		if warmup is None:
			warmup = steps / 2
		accepts = 0.0
		xs = []
		x = x0
		for step in range(steps):
			p0 = p(x)
			if proposal_param is None:
				xf = sample_q(x)
			else:
				xf = sample_q(x, proposal_param)
			pf = p(xf)
			odds = pf - p0
			if q is not None:
				if proposal_param is None:
					qf, qr = q(x, xf), q(xf, x)
				else:
					qf, qr = q(x, xf, proposal_param), q(xf, x, proposal_param)
				odds += qr - qf
			if proposal_param is not None and step < warmup:
				proposal_param = self._adapt_param(proposal_param, step, min(0, odds), target_rate)
			if np.log(npr.rand()) < odds:
				x = xf
				if step >= warmup:
					accepts += 1
			if step >= warmup and (step - warmup) % thin == 0:
				xs.append(x)
			accept_rate = accepts / (steps - warmup)
		return xs

	def _adapt_param(self, value, i, log_accept_prob, target_rate, const=3):
		"""
		Adapt the value of a parameter.
		"""
		new_val = value + const*(np.exp(log_accept_prob) - target_rate)/np.sqrt(i+1)
		# new_val = max(min_val, min(max_val, new_val))
		return new_val


	def mala(self, coreset, steps, thin, warmup):
		Z = coreset['coreset_data']
		D = coreset['coreset_dimension']
		w = coreset['coreset_weights']
		M = coreset['coreset_size']
		post_p = lambda theta, Z, w: (self.logistic_likelihood(theta, Z, w) - 0.5 * np.sum(theta**2) / 2.**2)
		def prop_mean(theta, ell, Z, w):
			grad = self._logistic_likelihood_grad(theta, Z, w)
			return theta + .5 * np.exp(2 * ell) * grad
		def q(theta, theta_new, ell, Z, w):
			diff = prop_mean(theta, ell, Z, w) - theta_new
			return -.5 * np.exp(-2 * ell) * np.sum(diff**2)
		def samp_q(theta, ell, Z, w):
			mean = prop_mean(theta, ell, Z, w)
			return mean + np.exp(ell) * np.random.randn(D)
		# initial state is 0
		theta0 = np.zeros(D)
		# initial adaptation parameter
		ell0 = np.log(2.38 / np.sqrt(D))	
		for i in range(int(M)):
			theta0 = self._mh(theta0,
			lambda theta: post_p(theta, Z, w),
			lambda theta, theta_new, ell: q(theta, theta_new, ell, Z, w),
			lambda theta, ell: samp_q(theta, ell, Z, w),
			steps=steps, warmup=warmup, thin=thin,
			proposal_param=ell0, target_rate=.574)[-1]

		return theta0