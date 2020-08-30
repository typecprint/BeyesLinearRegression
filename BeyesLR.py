import numpy as np
from scipy.stats import multivariate_normal


class BeyesLinearRegression:
    def __init__(self, mu, S, beta):
        self.mu = mu
        self.S = S
        self.beta = beta

    def calc_posterior(self, phi, t):
        S_inv = np.linalg.inv(self.S)

        if len(phi.shape) == 1:
            phi = phi.reshape(1, -1)
            t = t.reshape(1, 1)
        self.S = np.linalg.inv(S_inv + self.beta * phi.T @ phi)
        self.mu = self.S @ (S_inv @ self.mu + np.squeeze(self.beta * phi.T @ t))

    def sampling_params(self, n=1, random_state=0):
        np.random.seed(random_state)
        return np.random.multivariate_normal(self.mu, self.S, n)

    def probability(self, x):
        dist = multivariate_normal(mean=self.mu, cov=self.S)
        return dist.logpdf(x)

    def predict(self, phi):
        if len(phi.shape) == 1:
            phi = phi.reshape(1, -1)
        pred = np.array([self.mu.T @ _phi for _phi in phi])
        S_pred = np.array([(1 / self.beta) + _phi.T @ self.S @ _phi for _phi in phi])

        # Above is a simple implementation.
        # This may be better if you want speed.
        # pred = self.mu @ phi.T
        # S_pred = (1 / self.beta) + np.diag(phi @ self.S @ phi.T)
        return pred, S_pred
