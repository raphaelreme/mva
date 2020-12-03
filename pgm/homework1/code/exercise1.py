import argparse

import matplotlib.pyplot as plt
import numpy as np

import data
import lda
import linear
import logistic
import utils


methods = {
    "lda": lda.Lda,
    "logistic": logistic.Logistic,
    "linear": linear.Linear,
}

datasets = ["A", "B", "C"]


def main(directory, plot=True):
    for dataset in datasets:
        print("=======================================")
        print("Dataset", dataset)
        train_dataset, test_dataset =  data.load_dataset(dataset, directory)
        for method in methods:
            print("---------------------------------------")
            print("Method:", method)
            clf = methods[method]()
            clf.fit(*train_dataset)
            train_score = clf.score(*train_dataset)
            test_score = clf.score(*test_dataset)

            print("Train errors:", np.round(1 - train_score, 3))
            print("Test errors:", np.round(1 - test_score, 3))
            print("Learnt parameters:")
            w, b = np.round(clf.w, 3), np.round(clf.b, 3)
            print(f"w = ({w[0]}, {w[1]}),  b = {b}")
            if plot:
                plt.figure()
                utils.plot_decision_function(clf, train_dataset, test_dataset, dataset_name=dataset, show=False)
        print()

    if plot:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PGM Homework1")
    parser.add_argument(
        "-m",
        "--method",
        default="all",
        help="The method to use. [lda, linear, logistic]. Default to all.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="all",
        help="Dataset to use. [A, B, C]. Default to all",
    )
    parser.add_argument(
        "-D",
        "--data-directory",
        default="data",
        help="Directory where the dataset are stored. Default to ./data",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Will not plot the decision functions.",
    )

    args = parser.parse_args()
    if args.method != "all":
        methods = {args.method: methods[args.method]}
    if args.dataset != "all":
        datasets = [args.dataset]

    main(args.data_directory, not args.no_plot)
