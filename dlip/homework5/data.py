import numpy as np
import torch


def generate_trajectory(rossler, duration, start_from_equilibrium=False):
    """Generate a trajectory from a RosslerMap"""
    # Sample a random init point
    # Either near the equilibrium
    # Either in a random location in the approximate distribution

    if start_from_equilibrium:
        init_point = rossler.equilibrium() + np.random.normal(0, 0.01, 3)
    else:
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        z = np.random.uniform(0, 2)
        init_point = np.array([x, y, z])

    # init_point = torch.tensor([-5.75, -1.6, 0.02])

    traj, _ = rossler.full_traj(int(duration / rossler.delta_t), init_point)
    return traj


class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, rossler, n_traj, duration, p_eq=0.3):
        self.rossler = rossler
        self.duration = duration
        self.trajectories = torch.FloatTensor([
            generate_trajectory(rossler, duration, np.random.random() > 1 - p_eq) for _ in range(n_traj)]
        )

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]
