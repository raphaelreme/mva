import argparse
import random
import time

import matplotlib.pyplot as plt
import numpy as np


class ArmPicker:
    """Choose the arm to use"""
    def __init__(self, n_arms):
        self.N = np.zeros(n_arms)

    def __call__(self, R):
        i = self._next_arm(R)
        self.N[i] += 1
        return i

    def _next_arm(self, R):
        """Implement this method in subclasses to select the next arm to use."""


class UCBPicker(ArmPicker):
    def _next_arm(self, R):
        i = np.argmin(self.N)
        if self.N[i] == 0:
            return i

        t = sum(self.N) + 1
        B = R/self.N + np.sqrt(np.log(1 + t * np.log(t)**2 ) / (2*self.N))
        return np.argmax(B)


class KLUCBPicker(ArmPicker):
    @staticmethod
    def KL(mu_i, mu):
        if mu_i == 0:
            return -np.log(1 - mu)
        elif mu_i == 1:  # mu_i = 1 is never called
            return -np.log(mu)
        return mu_i * np.log(mu_i/mu) + (1 - mu_i) * np.log((1 - mu_i)/(1 - mu))

    def _next_arm(self, R):
        i = np.argmin(self.N)
        if self.N[i] == 0:
            return i

        t = sum(self.N) + 1
        B = np.log(1 + t * np.log(t)**2 ) / (2*self.N)

        mus = np.zeros(R.size)
        hat_mus = R/self.N

        for i in range(R.size):
            mu_i = hat_mus[i]

            # Bisection to find mu >= mu_i such that KL(mu_i, mu) = B
            # (As mu -> KL(mu_i, mu_i) increases on [mu_i, 1] from 0 to infinity)
            a = mu_i
            b = 1
            while b - a > 1e-7:
                c = (a + b) / 2
                d = self.KL(mu_i, c)
                if d > B[i]:
                    b = c
                else:
                    a = c

            mus[i] = a

        return np.argmax(mus)


def run(n, mus, picker):
    """Run an experiment over n time step

    Args:
        n (int): Total time
        mus (List[float]): Parameters of the r_i ~ B(mu_i)
        picker (str): The picker to use. ["UCB" | "KL-UCB"]

    returns:
        float: Sum of all the rewards during the experiment.
    """
    n_arms = len(mus)
    if picker == "UCB":
        picker = UCBPicker(n_arms)
    elif picker == "KL-UCB":
        picker = KLUCBPicker(n_arms)
    else:
        raise ValueError

    R = np.zeros(n_arms)
    for _ in range(n):
        i = picker(R)

        r = int(random.random() < mus[i])

        R[i] += r

    return sum(R)


def main(N, mu_1, n_delta):
    n = 10000
    deltas = [-0.5 + k/n_delta for k in range(n_delta + 1)]

    regrets = np.zeros((N, len(deltas), 2))
    times = []

    for i, delta in enumerate(deltas):
        t = time.time()
        mus = [mu_1, 0.5 + delta]
        optimum = n * max(mus)

        for j in range(N):
            regrets[j, i, 0] = optimum - run(n, mus, "UCB")

        for j in range(N):
            regrets[j, i, 1] = optimum - run(n, mus, "KL-UCB")

        times.append(time.time() - t)
        mean = sum(times)/len(times)
        print(f"TIME -> Mean: {mean:.3f}, Last: {times[-1]:.3f}, Remaining: {(len(deltas) - len(times)) * mean:.3f}           ", end="\r")

    mean_regrets = regrets.mean(axis=0)
    std_regrets = regrets.std(axis=0)
    plt.title(f"Expected regret w.r.t Delta (mu_1 = {mu_1})")
    plt.fill_between(deltas, mean_regrets[:,0] - std_regrets[:,0], mean_regrets[:,0] + std_regrets[:, 0], alpha=0.3, color="b")
    plt.fill_between(deltas, mean_regrets[:,1] - std_regrets[:,1], mean_regrets[:,1] + std_regrets[:, 1], alpha=0.3, color="r")
    plt.plot(deltas, mean_regrets[:, 0], color="b", label="UCB")
    plt.plot(deltas, mean_regrets[:, 1], color="r", label="KL-UCB")
    plt.xlabel("Delta (mu_2 = 1/2 + Delta)")
    plt.ylabel("Regret")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL assigment 3 - Exercise 3")
    parser.add_argument(
        "-N",
        type=int,
        default=30,
        help="Retry N times for each delta in order to have a better approximate of the expected regret.",
    )
    parser.add_argument(
        "--n-delta",
        type=int,
        default=30,
        help="Split the Delta space ([-0.5, 0.5]) in n_delta pieces.",
    )
    parser.add_argument(
        "--mu-1",
        type=float,
        default=0.5,
        help="First arm expectation.",
    )

    args = parser.parse_args()

    main(args.N, args.mu_1, args.n_delta)
