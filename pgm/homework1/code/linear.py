import numpy as np


class Linear:
    """Linear regression classifier

    Our statistical model is:
    y | x ~ N(wx + b, alpha) with alpha given.

    Solving the MLE is equivalent to solving the OLS:
    min_theta ||Y - X @ theta||_2 (with theta = (b, w.T))

    This class will find the optimum w, b solving the problem. And will be able to predict the class given x:
    y = (w.Tx +b > 0.5)

    Attrs:
        n (int): Number of samples
        p (int): Number of features
        w (array[p]), b (float): Parameters of the law of y given x. (Extract from pi, mu, sigma.)
    """
    def __init__(self):
        self.n = 0
        self.p = 0

        self.w = np.zeros(self.p)
        self.b = 0.

    def fit(self, X, Y):
        """Compute the optimum parameters w.r.t likelyhood for the dataset (X, Y)

        Args:
            X (array[n, p]): n samples of x
            Y (array[n]): n samples of y

        Returns:
            Logistic: self
        """
        self.n, self.p = X.shape

        X_ = np.hstack((np.ones((self.n, 1)), X))

        theta = np.linalg.lstsq(X_, Y, rcond=None)[0]

        self.w = theta[1:]
        self.b = theta[0]

        return self

    def predict(self, X):
        """Predict the most probable class.

        Args:
            X (array[n, p]): n samples to classify

        Returns:
            array[n]: Predicted classes
        """
        return X @ self.w + self.b > 0.5

    def score(self, X, Y):
        """Compute the score of the classifier on the dataset (X, Y)

        Args:
            X (array[n, p]): n samples to classify
            Y (array[n]): Ground truth

        Returns:
            float: Proportion of well classified samples
        """
        return np.mean(Y == self.predict(X))

    def decision_function(self, x):
        """Decision function for 2D examples.

        Will return z = f(x) such that p(y=1|(x, z)) = 0.5.

        Args:
            x (array[m]): First coordinates.

        Returns:
            z (array[m]): The decision function at x.
        """
        return - self.w[0]/self.w[1] * x - self.b/self.w[1] + 0.5/self.w[1]
