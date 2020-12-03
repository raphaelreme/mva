import numpy as np

from utils import sigmoid


class Lda:
    """LDA classifier

    Our statistical model is:
    y ~ B(pi), x | y = i ~ N(mu_i, sigma)

    Will solve the MLE problem with this model. And able to predict the probability for x to be in class i:

    p(y=1|x) = sigmoid(wx + b)

    Attrs:
        n (int): Number of samples
        p (int): Number of features
        pi (float), mu_0 (array[p]), mu_1 (array[p]), sigma (array[p,p]): Parameters of the model to learn
        w (array[p]), b (float): Parameters of the law of y given x. (Extract from pi, mu, sigma.)
    """
    def __init__(self):
        self.n = 0
        self.p = 0

        self.pi = 0.
        self.mu_0 = np.zeros(self.p)
        self.mu_1 = np.zeros(self.p)
        self.sigma = np.zeros((self.p, self.p))

        self.w = np.zeros(self.p)
        self.b = 0.

    def fit(self, X, Y):
        """Compute the optimum parameters w.r.t likelyhood for the dataset (X, Y)

        Args:
            X (array[n, p]): n samples of x
            Y (array[n]): n samples of y

        Returns:
            LDA: self
        """
        self.n, self.p = X.shape

        X_0 = X[Y == 0]
        X_1 = X[Y == 1]

        self.pi = np.mean(Y)
        self.mu_0 = np.mean(X_0, axis=0)
        self.mu_1 = np.mean(X_1, axis=0)
        self.sigma = ((X_1 - self.mu_1).T @ (X_1 - self.mu_1)) + ((X_0 - self.mu_0).T @ (X_0 - self.mu_0))
        self.sigma /= self.n

        sigma_inv = np.linalg.inv(self.sigma)
        self.w = sigma_inv @ (self.mu_1 - self.mu_0)
        self.b = self.mu_0.T @ sigma_inv @ self.mu_0 - self.mu_1.T @ sigma_inv @ self.mu_1
        self.b = self.b * 0.5 + np.log(self.pi/(1 - self.pi))

        return self

    def predict(self, X, mode="probability"):
        """Predict p(y=1|x) or the most probable class.

        Args:
            X (array[n, p]): n samples to classify
            mode ("probability"|"class"): In probability mode returns p(y=1|x)
                In class mode, returns the most probable class

        Returns:
            array[n]: Predictions. (See `mode`)
        """
        probs = sigmoid(X @ self.w + self.b)
        if mode == "probability":
            return probs
        elif mode == "class":
            return probs > 0.5
        else:
            raise ValueError("Invalid mode")

    def score(self, X, Y):
        """Compute the score of the classifier on the dataset (X, Y)

        Args:
            X (array[n, p]): n samples to classify
            Y (array[n]): Ground truth

        Returns:
            float: Proportion of well classified samples
        """
        return np.mean(Y == self.predict(X, mode="class"))

    def decision_function(self, x):
        """Decision function for 2D examples.

        Will return z = f(x) such that p(y=1|(x, z)) = 0.5.

        Args:
            x (array[m]): First coordinates.

        Returns:
            z (array[m]): The decision function at x.
        """
        return - self.w[0]/self.w[1] * x - self.b/self.w[1]
