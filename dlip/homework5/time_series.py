import argparse
from scipy.interpolate import interp1d
import numpy as np
import torch

from model import LinearModel

parser = argparse.ArgumentParser()
parser.add_argument("--init", nargs="+", type=float,  default=[-5.75, -1.6, 0.02])
value = parser.parse_args()


class RosslerModel:
    def __init__(self, delta_t):
        self.delta_t = delta_t #if discrete model your delta_t
                              #if continuous model chose one <=1e-2
        self.nb_steps = int(10000 / self.delta_t)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.rosler_nn = LinearModel((3, 50, 100, 200, 3)).to(self.device)
        self.rosler_nn.load_state_dict(torch.load("best.pth", map_location=self.device))
        self.rosler_nn.to(self.device)  # Ensure again to be on the right device

    def full_traj(self, initial_condition): 
        # run your model to generate the time series with nb_steps
        # just the y cordinate is necessary.
        traj = [initial_condition]
        x = torch.FloatTensor(initial_condition).to(self.device)

        with torch.no_grad():
            for i in range(self.nb_steps):
                x = self.rosler_nn(x)
                traj.append(x.cpu().numpy())

        traj = np.array(traj)
        y = traj[:, 1]

        t = np.linspace(0, 10000, self.nb_steps + 1)
        t_new = np.linspace(0, 10000, 10**6 + 1)
        # Drop last t as we don't know what to do with it
        # (t=[0, 10000] with dt = 0.01 => 1e6 + 1 points)
        y_new = interp1d(t, y)(t_new)[:-1]

        #if your delta_t is different to 1e-2 then interpolate y
        #in a discrete time array t_new = np.linspace(0,10000, 10000//1e-2)
        # y_new = interp1d(t_new, your_t, your_y)
        # I expect that y.shape = (1000000,)
        return y_new

    def save_traj(self, y):
        #save the trajectory in traj.npy file
        # y has to be a numpy array: y.shape = (1000000,)
        np.save('traj.npy', y)
        
    
if __name__ == '__main__':
    ROSSLER = RosslerModel(0.1)

    y = ROSSLER.full_traj(np.array(value.init))

    ROSSLER.save_traj(y)
