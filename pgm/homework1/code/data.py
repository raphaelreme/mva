import numpy as np
import pyreadr


def load_dataset(dataset, directory):
    X_train, Y_train = read_data(f"{directory}/train{dataset}")
    X_test, Y_test = read_data(f"{directory}/test{dataset}")
    return (X_train, Y_train), (X_test, Y_test)


def read_data(filename):
    """Read the data file and returns X, Y

    Based on the format of the files of this homework. (Ex 1)

    Expected format:
    'X[0, 0] X[0, 1] .... X[0, d-1] "Y[0]"
    ...
    "X[0, 0] X[0, 1] .... X[0, d-1] "Y[0]"
    ...
    X[n-1, 0] X[n-1, 1] ... X[n-1, d-1] "Y[n-1]"'

    Returns:
        array[n, p], array[n]: X, Y
    """
    X = []
    Y = []
    with open(filename, "r") as file:
        data = file.read().strip().split("\n")
    for line in data:
        line = line.strip().split(" ")
        X.append([])
        for elt in line[:-1]:
            X[-1].append(float(elt))
        Y.append(int(line[-1][1:-1]))
    return np.array(X), np.array(Y)


def read_decathlon_RData(filename):
    """Read the file decathlon.RData for ex2.

    Uses pyreadr to do it.

    Args:
        filename (str): Read from this file

    Returns:
        list[p]: Column names
        array[n, p]: Dataset
    """
    df = pyreadr.read_r(filename)["X"]
    return df.columns.to_list(), df.to_numpy()
