import numpy as np
import scipy.optimize

from utils import sigmoid


class Logistic:
    """Logistic regression classifier

    Our statistical model is:
    y | x ~ B(sigmoid(wx + b))

    Will solve the MLE problem with this model. And able to predict the probability for x to be in class i:

    p(y=1|x) = sigmoid(wx + b)

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

    @staticmethod
    def grad_log_likelyhood(theta, X, Y):
        """Compute the gradient of the log likelyhood.

        Args:
            theta (array[p+1]): (b, w.T)
            X (array[n, p+1]): n samples of (1, x.T)
            Y (array[n]): n samples of y

        Returns:
            array[p+1]: The gradient of the log-likelyhood at theta.
        """
        return X.T @ (Y - sigmoid(X @ theta))

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

        theta, _, success, msg = scipy.optimize.fsolve(
            self.grad_log_likelyhood,
            np.zeros(self.p + 1),
            args= (X_, Y),
            full_output=True,
        )

        if success != 1:
            print(msg)
            assert np.allclose(np.zeros(self.p + 1), self.grad_log_likelyhood(theta, X_, Y))
            print("Still a good approximation. Let's go on.")

        self.w = theta[1:]
        self.b = theta[0]

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
