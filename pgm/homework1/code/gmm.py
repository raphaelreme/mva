import itertools

import numpy as np
from sklearn.cluster import KMeans


def _multi_normal(X, mu, sigma):
    """Compute at once the density function at points X for the all the normals distribution
    of parameter mu, sigma.

    Args:
        X  (array[n, p]): Points to evaluate
        mu (array[K, p]): Means of the normal distributions
        sigma (array[K, p, p]): Covariances of the normal distributions

    Returns:
        array[n, K]: The pdf at X[i] of N(mu[k], sigma[k])
    """
    deviation = X[:, None, :] - mu  # Shape (K, n, p)
    sigma_inv = np.linalg.inv(sigma)  # Shape (K, p, p)
    pdf = (deviation[..., None, :] @ sigma_inv @ deviation[..., None]).squeeze()  # Shape (n, K)
    pdf = np.exp(-0.5 * pdf)  # Shape (n, K)
    fact = np.sqrt(np.abs(np.linalg.det(sigma)) * (2 * np.pi)**X.shape[1])  # shape (K,)
    return pdf / fact


class Gmm:
    """Gaussian mixture model classifier

    Our statistical model is:
    x ~ \sum_k pi_k N(mu_k, sigma_k)

    Which is equivalent to introduce latent variable z (which will be the class of x):
    z ~ M(1, pi)
    x | z = e_k ~ N(mu_k, sigma_k)

    Will solve the MLE problem with an EM algorithm. And able to predict the probability for x to be of class k:
    p(z=e_k|x) = tau_k

    Attrs:
        K (int): Number of classes
        n (int): Number of samples
        p (int): Number of features
        pi    (array[K]): Current estimation of pi
        mu    (array[K, p]): Current estimations of mu
        sigma (array[K, p, p]): Current estimations of sigma
        tau   (array[n, K]): Current estimation of p(z=e_k|x)
        init ("kmeans"|"random"): Use kmeans to initialized tau, or randomly sample the classes of each points
        precision (float): Stops EM as soon as the algorithm does not improve the log-likelyhood more than precision
        covar_regulation (float): Insure that the covariances are positive. (really needed for the decathlon dataset.)
    """
    def __init__(self, K, init="kmeans", precision=1e-10, covar_regulation=1e-6):
        self.K = K
        self.n = 0
        self.p = 0

        assert init in ["kmeans", "random"], "Invalid init argument"
        self.init = init
        self.precision = precision
        self.covar_regulation = covar_regulation

        self._log_likelyhood_evolution = []  # Keep tracks of the convergence evolution during fit

        self.tau = np.zeros((self.n, self.K))
        self.pi = np.zeros(self.K)
        self.mu = np.zeros((self.K, self.p))
        self.sigma = np.zeros((self.K, self.p, self.p))

    def copy(self):
        """Copy itself

        returns:
            Gmm: The copy
        """
        copy = Gmm(self.K, self.init, self.precision, self.covar_regulation)
        copy.n, copy.p = self.n, self.p
        copy._log_likelyhood_evolution = self._log_likelyhood_evolution.copy()
        copy.tau = self.tau
        copy.pi = self.pi
        copy.mu = self.mu
        copy.sigma = self.sigma
        return copy

    def _init(self, X):
        """Initialized the all the parameters before EM.

        Args:
            X (array[n, p]): The dataset to fit

        Returns:
            Gmm: self
        """
        self.n, self.p = X.shape
        assert self.K <= self.n, "More class than samples..."

        if self.init == "kmeans":
            classes = KMeans(n_clusters=self.K).fit_predict(X)
        else:
            classes = np.random.randint(0, self.K, self.n)
            for k in range(self.K):
                classes[k] = k  # Ensure that \pi_k != 0 or we will be stuck!
            np.random.shuffle(classes)

        self.tau = np.eye(self.K)[classes]
        self._M_step(X)  # Init pi, mu, sigma.

        return self

    def _E_step(self, X):
        """Performs E step of the Guassian mixture model:

        Update tau such that R = \prod_i M(1, tau_i) maximizes \mathcal{L}

        Args:
            X (array[n, p]): The dataset to fit

        Returns:
            Gmm: self
        """
        self.tau = _multi_normal(X, self.mu, self.sigma) * self.pi  # Shape (n, K)
        self.tau /= self.tau.sum(axis=1, keepdims=True)  # Shape (n, K)
        return self

    def _M_step(self, X):
        """Perform M step of the Guassian mixture model:

        Update pi, mu, sigma such that it maximizes \mathcal{L}

        Args:
            X (array[n, p]): The dataset to fit

        Returns:
            Gmm: self
        """
        N = self.tau.sum(axis=0)  # Shape (K,)
        self.pi = N/self.n  # Shape (K,)
        self.mu = 1/N[:, None] * (self.tau.T @ X)  # Shape (K, p)
        deviation = X[None, ...] - self.mu[:, None, ]  # Shape (K, n, p)
        self.sigma = deviation.transpose(0, 2, 1) @ (self.tau.T[..., None] * deviation)  # Shape (K, p, p)
        self.sigma /= N[:, None, None]  # Shape (K, p, p)
        self.sigma += self.covar_regulation * np.eye(self.p)

        return self

    def log_likelyhood(self, X):
        """Compute the log-likelyhood at X of our estimated parameter.

        L(X, theta) = \sum_i log \sum_k \pi_k N(X_i; \mu_k, \sigma_k)

        Args:
            X (array[n, p]): The dataset to fit

        Returns:
            float: The log-likelyhood at X
        """
        A = _multi_normal(X, self.mu, self.sigma) * self.pi  # Shape (n, K)
        return np.sum(np.log(A.sum(axis=1)))

    def fit(self, X, *args):
        """EM algorithm to compute the parameter of our model.

        Args:
            X (array[n, p]): The dataset to fit

        Returns:
            Gmm: self
        """
        self._init(X)

        self._log_likelyhood_evolution = [self.log_likelyhood(X)]

        iterations = 0
        while True:
            iterations += 1
            theta = (self.pi, self.mu, self.sigma)
            self._E_step(X)._M_step(X)

            self._log_likelyhood_evolution.append(self.log_likelyhood(X))
            # Could check the loglikelyhood but when it does not converge we have some issues with it...
            # if self._log_likelyhood_evolution[-1] < self._log_likelyhood_evolution[-2] + self.precision:
            #     if self._log_likelyhood_evolution[-1] < self._log_likelyhood_evolution[-2]:
            #         print("Warning, the likelyhood has decreased")
            #     break

            # Rather check that the parameters converged. But it's not always the case. Add also a check on the iterations.
            if np.allclose(self.pi, theta[0]) and np.allclose(self.mu, theta[1]) and np.allclose(self.sigma, theta[2]):
                break

            if iterations > 100:
                break

        return self

    def predict(self, X, mode="probability"):
        """Predict p(z|x) or the most probable class.

        Args:
            X (array[n, p]): n samples to classify
            mode ("probability"|"class"): In probability mode returns p(z|x)
                In class mode, returns the most probable class

        Returns:
            array[n, K]: For mode = "probability": p(z=e_k|x_i)
            Or
            array[n]: For mode = class: Most probable class for each x_i
        """
        tau = _multi_normal(X, self.mu, self.sigma) * self.pi  # Shape (n, K)
        tau /= tau.sum(axis=1, keepdims=True)  # Shape (n, K)

        if mode == "probability":
            return tau
        elif mode == "class":
            return tau.argmax(axis=1)
        else:
            raise ValueError("Invalid mode")

    def score(self, X, Y):
        """Compute the score of the classifier on the dataset (X, Y)

        Try all permutations of classes and returns the score found.
        WARNING: Will not work for big K.

        Args:
            X (array[n, p]): n samples to classify
            Y (array[n]): Ground truth

        Returns:
            float: Proportion of well classified samples
        """
        classes = np.unique(Y)
        Y_pred = self.predict(X, mode="class")  # Each pred is in [0, K-1]
        if len(classes) < self.K:
            Y_pred = np.min((Y_pred, (len(classes) - 1) * np.ones(len(Y), dtype=int)), axis=0)  # Want to predict less classes... Let's merge the last clusters

        max_score = 0
        for permutation in itertools.permutations(classes):
            permutation = np.array(permutation)
            Y_pred_perm = permutation[Y_pred]
            max_score = max(np.mean(Y == Y_pred_perm), max_score)
        return max_score
