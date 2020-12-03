import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

import data
import gmm
import utils


def visual(directory, K, init):
    for dataset in ["A", "B", "C"]:
        print("=======================================")
        print("Dataset", dataset)
        train_dataset, test_dataset =  data.load_dataset(dataset, directory)

        print("---------------------------------------")
        print(f"Method: GMM")
        clf = gmm.Gmm(K=K, init=init)
        clf.fit(train_dataset[0])
        print("Converged in", len(clf._log_likelyhood_evolution), "iterations")
        print(clf._log_likelyhood_evolution)
        train_score = clf.score(*train_dataset)
        test_score = clf.score(*test_dataset)

        print("Train errors:", np.round(1 - train_score, 3))
        print("Test errors:", np.round(1 - test_score, 3))
        print()
        plt.figure()
        utils.plot_decision_function(clf, train_dataset, test_dataset, dataset_name=dataset, show=False)

    plt.show()


def decathlon(directory, K, init):
    print("=======================================")
    print(f"Dataset: Decathlon, Method: GMM")
    columns, X = data.read_decathlon_RData(f"{directory}/decathlon.RData")

    clf = gmm.Gmm(K=K, init=init)
    clf.fit(X)
    # tmp = gmm.Gmm(K=K, init=init)
    # for k in range(50):
    #   tmp.fit(X):
    #   if tmp better than clf:
    #       clf = tmp.copy()
    # But hasn't improve really things... So let's discard it.

    print("EM converged in", len(clf._log_likelyhood_evolution), "iterations")

    Y = clf.predict(X, mode="class")
    for k in range(K):
        print("----------------------------------------------------------")
        print(f"Mean cluster {k}:")
        print(pd.DataFrame(np.round(clf.mu[k], 2)[None,:], columns=columns))
        print("----------------------------------------------------------")
    print()
    for k in range(K):
        print("----------------------------------------------------------")
        print(f"Cluster {k}:")
        print(pd.DataFrame(X, columns=columns)[Y==k])
        print("----------------------------------------------------------")

    plt.figure()
    plt.title("Convergence of EM algorithm")
    plt.xlabel("Iterations")
    plt.ylabel("Log-Likelyhood")
    plt.ylim(np.round(np.min(clf._log_likelyhood_evolution) - 2), np.round(np.max(clf._log_likelyhood_evolution) + 3))
    plt.plot(clf._log_likelyhood_evolution, label="Log-likelyhood")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PGM Homework1")
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Use GMM on 2D datasets A, B, C (By default it will use GMM for the decathlon dataset)"
    )
    parser.add_argument(
        "-D",
        "--data-directory",
        default="data",
        help="Directory where the dataset are stored. Default to ./data",
    )
    parser.add_argument(
        "-K",
        "--clusters",
        type=int,
        default=2,
        help="Number of clusters we should look for",
    )
    parser.add_argument(
        "-i",
        "--init",
        default="kmeans",
        help="Initialization method for EM algorithm",
    )

    args = parser.parse_args()

    if args.visual:
        visual(args.data_directory, args.clusters, args.init)
    else:
        decathlon(args.data_directory, args.clusters, args.init)
