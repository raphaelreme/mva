import argparse
import enum

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


CONVERGENCE_TEST = False


class Direction(enum.IntEnum):
    UP=0
    DOWN=1
    RIGHT=2
    LEFT=3


class LoopyBeliefPropagation:
    """Run the loopy belief Propagation algorithm in our Ising configuration (Sum-Product)

    We have p(x) = 1/Z \prod_i phi_i(x_i) \prod_{i,j \in E} \psi_{i,j}(x_i, x_j).

    With phi_i(x) = exp(alpha * x_i - 0.5 * (y_i - x_i)**2)
    And psi_{i,j}(x_i, x_j) = psi(x_i, x_j) = exp(beta * (x_i == x_j)) It does not depend on i, j.

    Using that we are manipulating images, let's call H, W the dimension of the image

    With h, w, phi can be compute as an array of size (H, W, 2). (2 possibility for each x_{h, w})
    And psi can be stored simply as: [exp(0), exp(beta)]

    mu the incoming messages at the pixel (h, w) can be compute for each directions: (only computed for neighbors)
    mu[dir, h, w, x] = \sum_{x'} phi[h', w', x'] psi[x == x'] \prod_{dir'!=-dir} mu[dir', h', w', x'] 
    (with (h', w') the neighboring pixel in the direction dir. If there is None, mu is kept to 1)

    Attrs:
        H (int): Height of the image
        W (int): Width of the image
        phi (array[H, W, 2]): Potential for each pixel
        psi (array[2]): Potential for each neighboring pixel (which is given by psi[x == x'])
        mu (array[4, H, W, 2]): Messages for each direction in each pixels.
    """
    alpha = 0
    beta = 1

    def __init__(self, y, mu):
        """Constructor.

        Args:
            y (array[H, W]): Observed image
            mu (array[2]): Gaussian means of the noise
        """
        self.H, self.W = y.shape
        self._x_range = np.array([0, 1])

        self.phi = np.exp(self.alpha * self._x_range - 0.5*(y[..., None] - mu[self._x_range])**2)
        self.psi = np.exp(self.beta * self._x_range)

        self.mu = np.ones((4, self.H, self.W, self._x_range.size))
        self._new_mu = np.ones((4, self.H, self.W, self._x_range.size))
        self._prod_mu = np.ones((self.H, self.W, self._x_range.size))

    def compute_messages(self, N):
        """Compute all the messages in N iterations

        For each pixel and direction, it will recompute a new mu from the previous one.

        Args:
            N (int): Max number of iterations
        """
        conv_mu = []
        conv_belief = []
        for i in range(N):
            print(f"Iterations: {i+1}/{N}...", end="            \r")
            self._new_mu = np.ones((4, self.H, self.W, self._x_range.size))
            for dir in [Direction.UP, Direction.DOWN, Direction.RIGHT, Direction.LEFT]:  # msg from dir
                for w in range(self.W):
                    w_p = w
                    if dir == Direction.RIGHT:
                        w_p += 1
                    elif dir == Direction.LEFT:
                        w_p -= 1
                    if w_p < 0 or w_p >= self.W:
                        continue  # Don't take direction that leads out of the image

                    for h in range(self.H):
                        h_p = h
                        if dir == Direction.UP:
                            h_p -= 1
                        elif dir == Direction.LEFT:
                            h_p += 1
                        if h_p < 0 or h_p >= self.H:
                            continue  # Don't take direction that leads out of the image

                        # For each valide direction and pixel, compute the new mu
                        self._update(dir, w, h, w_p, h_p)

            if CONVERGENCE_TEST:
                conv_mu.append(np.linalg.norm((self._new_mu - self.mu).ravel(), ord=1))
                conv_belief.append(
                    np.linalg.norm((self.phi * (np.prod(self.mu, axis=0) - np.prod(self._new_mu, axis=0))).ravel(), ord=1)
                )

            self.mu = self._new_mu
            self._prod_mu = np.prod(self.mu, axis=0)

        print()
        if CONVERGENCE_TEST:
            print("Absolute diff in mu through iterations:\n", np.array(conv_mu))
            print("Absolute diff in belief through iterations:\n", np.array(conv_belief))

    def _update(self, dir, w, h, w_p, h_p):
        opposite_dir = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.RIGHT: Direction.LEFT,
            Direction.LEFT: Direction.RIGHT,
        }[dir]

        self._new_mu[dir, h, w, :] = 0
        for x in self._x_range:
            for x_p in self._x_range:
                self._new_mu[dir, h, w, x] += self.phi[h_p, w_p, x_p] * self.psi[int(x == x_p)] * self._prod_mu[h_p, w_p, x_p]/self.mu[opposite_dir, h_p, w_p, x_p]

        # Normalizes the messages to avoid that they overflow.
        self._new_mu[dir, h, w, :] /= np.sum(self._new_mu[dir, h, w, :])

    def p(self, pos=None):
        """Compute the a-posteriory probability of x | y

        Can be run after that `compute_messages` has been called.

        Args:
            pos (Optional[Tuple[int, int]]): h, w = pos. Return only p(x_pos | y)

        Returns:
            array[W, H, 2]: p(x_pos | y) for all pos (if no pos is given)
            OR
            array[2]: p(x_pos | y) if pos is given
        """
        if pos:
            h, w = pos
            return self.phi[h, w] * np.prod(self.mu[:, h, w, :], axis=0)
        else:
            return self.phi * np.prod(self.mu, axis=0)


def load_image(file):
    """Load an black/white image as an array

    Args:
        file (str): Image location

    Returns:
        array[H, W]: Binary image
    """
    image = Image.open(file).convert("1")
    image = np.array(image).astype(np.int8)
    return image


def noisy(image, mu):
    """Generate a noisy binary image with gaussians.

    The image generated is y_i | x_i = l ~ N(mu[l], 1) with thresholding.

    Args:
        image (array[H, W]): Input binary image
        mu (array[2]): The two gaussian means

    Returns:
        array[H, W]: The noised image.
    """
    # noise_0 = (np.random.normal(mu[0], 1, image.shape) > 0.5).astype(np.int8)
    # noise_1 = (np.random.normal(mu[1], 1, image.shape) > 0.5).astype(np.int8)
    noise_0 = np.random.normal(mu[0], 1, image.shape)
    noise_1 = np.random.normal(mu[1], 1, image.shape)
    return noise_0 * (image == 0) + noise_1 * (image == 1)


def main(img_file, mu, N):
    x = load_image(img_file)
    y = noisy(x, mu)

    LBP = LoopyBeliefPropagation(y, mu)
    LBP.compute_messages(N)

    x_p = np.argmax(LBP.p(), axis=2)

    plt.figure()
    plt.title("Initial image")
    plt.axis("off")
    plt.imshow(x, cmap="gray")

    plt.figure()
    plt.title("Noised image")
    plt.axis("off")
    plt.imshow((y > 0.5).astype(np.int8), cmap="gray")

    plt.figure()
    # Note; This denoised version does not optimize p(x | y) but only all the p(x_i | y) separately.
    plt.title("Denoised image")
    plt.axis("off")
    plt.imshow(x_p, cmap="gray")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PGM Homework2")
    parser.add_argument(
        "image",
        help="Location of the binary image.",
    )
    parser.add_argument(
        "--mu-0",
        type=float,
        default=0.2,
        help="Mean of the gaussian noise for black (0) pixels. Default: 0.2",
    )
    parser.add_argument(
        "--mu-1",
        type=float,
        default=0.8,
        help="Mean of the gaussian noise for white (1) pixels. Default: 0.8",
    )
    parser.add_argument(
        "-N",
        type=int,
        default=5,
        help="Number of iterations. Default: 5",
    )
    parser.add_argument(  # It does not converge enough to stop the algorithm with a convergence criteria
        "--check-convergence",
        action="store_true",
        help="Debugging option to check mu/belief convergence.",
    )
    args = parser.parse_args()

    if args.check_convergence:
        CONVERGENCE_TEST = True

    mu = np.array([args.mu_0, args.mu_1])
    main(args.image, mu, args.N)
