import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plot_decision_function(clf, train_dataset, test_dataset=None, dataset_name=None, show=True):
    """Plot the decision function learn for the classifier w.r.t the given dataset

    Args:
        clf: Classifier
        train_dataset (Tuple[array[n, p], array[n]]): Dataset to train the classifier
        test_dataset (Tuple[array[n', p], array[n]): Test data set (To see how well the classifier performs)
        dataset_name (str): For the plot title.
    """
    colors = np.array(['#1f77b4', '#ff7f0e', "r"])  # Defaut blue/orange of matplotlib + r for separation line.

    X_train, Y_train = train_dataset
    plt.scatter(X_train[:,0], X_train[:, 1], c=colors[Y_train])
    for klass in np.unique(Y_train):
        plt.scatter([], [], c=colors[klass], label=f"Train - Class {klass}")

    # Plot decision boundary
    N = 50  # Division of the space in N points.

    x_min, x_max = np.min(X_train[:,0]), np.max(X_train[:,0])
    x_min, x_max = x_min - 0.2 * (x_max - x_min), x_max + 0.2 * (x_max - x_min)
    x = np.linspace(x_min, x_max, N)

    decision_function = getattr(clf, "decision_function", None)
    if callable(decision_function):  # If decision_function is implemented, it's easy.
        plt.plot(x, decision_function(x), c=colors[2])
    else:  # Else let's try to find the boundary in the 2D space where p(y=1|x) > 0.5
        y_min, y_max = np.min(X_train[:,1]), np.max(X_train[:,1])
        y_min, y_max = y_min - 0.2 * (y_max - y_min), y_max + 0.2 * (y_max - y_min)
        y = np.linspace(y_min, y_max, N)  # Shape (N,)
        x, y = np.meshgrid(x, y)  # Shape (N, N), (N, N)
        X = np.dstack((x, y)).reshape(-1, 2) # Shape (N*N, 2)
        z = clf.predict(X)  # Shape (N*N,) or (N*N, 2) for gmm
        if len(z.shape) == 2:
            z = z[:, 1]  # Keep only p(y=1|x)
        z = z.reshape(x.shape)  # Shape (N, N)

        plt.contour(x, y, z, colors=colors[2], levels=[0.5], linestyles='-')


    if test_dataset is not None:
        X_test, Y_test = test_dataset
        plt.scatter(X_test[:,0], X_test[:, 1], c=colors[Y_test], alpha=0.4)
        for klass in np.unique(Y_test):
            plt.scatter([], [], c=colors[klass], alpha= 0.4, label=f"Test - Class {klass}")

    if dataset_name:
        plt.title(f"{clf.__class__.__name__} classification over dataset {dataset_name}")
    else:
        plt.title(f"{clf.__class__.__name__} classification")
    plt.legend()
    if show:
        plt.show()
