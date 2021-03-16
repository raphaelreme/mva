import argparse

import numpy as np
from numpy.linalg import qr
import torch

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from rossler_map import RosslerMap
from model import LinearModel
from data import generate_trajectory


def main(args):
    np.random.seed(42)
    rossler = RosslerMap(delta_t=0.1)
    true_traj = generate_trajectory(rossler, args.duration, args.equilibrium)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = LinearModel((3, 50, 100, 200, 3)).to(device)
    model.load_state_dict(torch.load(args.save_file, map_location=device))
    model.to(device)  # Ensure again to be on the right device

    pred_traj = [true_traj[0]]
    x = torch.FloatTensor(pred_traj[0]).to(device)
    with torch.no_grad():
        for i in range(len(true_traj) - 1):
            x = model(x)
            pred_traj.append(x.cpu().numpy())
    pred_traj = np.array(pred_traj)

    if args.equilibrium:
        plt.figure()
        plt.title("Y predicted near equilibrium")
        plt.plot(pred_traj[:, 1])

        plt.figure()
        plt.title("True Y near equilibrium")
        plt.plot(true_traj[:, 1])


    # Trajectories
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.title("Trajectories")
    ax.plot(pred_traj[:,0], pred_traj[:,1], pred_traj[:,2], alpha=0.5, label="Predicted")
    ax.plot(true_traj[:,0], true_traj[:,1], true_traj[:,2], alpha=0.5, label="True")
    ax.legend()

    # PDFs
    fig = plt.figure()
    plt.title("X repartition")
    plt.hist(pred_traj[:, 0], alpha=0.5, label="Predicted", bins=range(-10, 10, 2))
    plt.hist(true_traj[:, 0], alpha=0.5, label="True", bins=range(-10, 10, 2))
    plt.legend()

    fig = plt.figure()
    plt.title("Y repartition")
    plt.hist(pred_traj[:, 1], alpha=0.5, label="Predicted", bins=range(-10, 10, 2))
    plt.hist(true_traj[:, 1], alpha=0.5, label="True", bins=range(-10, 10, 2))
    plt.legend()

    fig = plt.figure()
    plt.title("Z repartition")
    plt.hist(pred_traj[:, 2], alpha=0.5, label="Predicted", bins=range(10))
    plt.hist(true_traj[:, 2], alpha=0.5, label="True", bins=range(10))
    plt.legend()

    # Autocorrelation
    fig = plt.figure()
    plt.title("Auto-correlation (First 1000 coefficients)")
    plt.plot(autocorrelation(pred_traj[:, 1])[:1000], alpha=0.5, label="Predicted")
    plt.plot(autocorrelation(true_traj[:, 1])[:1000], alpha=0.5, label="True")
    plt.legend()

    # Fourier transforms
    fig = plt.figure()
    plt.title("Fourier transform (First 1000 coefficients)")
    plt.plot(np.abs(np.fft.rfft(pred_traj[:, 1]))[:1000], alpha=0.5, label="Predicted")
    plt.plot(np.abs(np.fft.rfft(true_traj[:, 1]))[:1000], alpha=0.5, label="True")
    plt.legend()

    # Find fix point
    print("FIX POINT ANALYSYS")
    def f(x):
        return x - model(x)

    equilibirum = torch.FloatTensor(rossler.equilibrium()).to(device)
    v = f(equilibirum).cpu().detach().numpy()
    fix_point = newton(f,  torch.FloatTensor(true_traj[0]).to(device), device).cpu().detach().numpy()
    equilibirum = equilibirum.cpu().numpy()

    print(f"Equilibrium: ({equilibirum[0]:.3f}, {equilibirum[1]:.3f}, {equilibirum[2]:.3f})")
    print(f"Variation at equilibrium: ({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}) (Should be 0)")
    print(f"Found Fix Point: ({fix_point[0]:.3f}, {fix_point[1]:.3f}, {fix_point[2]:.3f})")
    print(f"Errors: {np.linalg.norm(fix_point - equilibirum).item()}")

    lyap = lyapunov_exponent(pred_traj, model, device, max_it=len(pred_traj))
    print("Lyapunov Exponents :", lyap, "with delta t =", 0.1)

    plt.show()


def autocorrelation(x):
    corr = np.correlate(x, x, mode="full")
    corr = corr[x.shape[0] - 1:]
    corr /= corr[0]
    return corr


def newton(f, x, device):
    tol = 1
    while tol>1e-5:
        x_old = x
        J_x = torch.autograd.functional.jacobian(f, x).detach().cpu().numpy()
        f_x = f(x).detach().cpu().numpy()
        x = x - torch.FloatTensor(np.linalg.solve(J_x, f_x)).to(device)
        tol = torch.norm(x - x_old).item()
    return x


def lyapunov_exponent(traj, model, device, max_it=1000, delta_t=0.1):
    n = traj.shape[1]
    w = np.eye(n)
    rs = []
    chk = 0

    for i in range(max_it):
        J = torch.autograd.functional.jacobian(model, torch.FloatTensor(traj[i]).to(device))
        J = J.detach().cpu().numpy()
        A = (J - np.eye(n)) / delta_t
        #WARNING this is true for the jacobian of the continuous system!
        w_next = np.dot(J, w) 
        #if delta_t is small you can use:
        #w_next = np.dot(np.eye(n)+jacob * delta_t,w)

        w_next, r_next = qr(w_next)

        # qr computation from numpy allows negative values in the diagonal
        # Next three lines to have only positive values in the diagonal
        d = np.diag(np.sign(r_next.diagonal()))
        w_next = np.dot(w_next, d)
        r_next = np.dot(d, r_next.diagonal())

        rs.append(r_next)
        w = w_next
        if i//(max_it/100)>chk:
            print(i//(max_it/100))
            chk +=1

    return  np.mean(np.log(rs), axis=0) / delta_t


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=1000)
    parser.add_argument("--save-file", default="best.pth")
    parser.add_argument("--equilibrium", action="store_true")
    args = parser.parse_args()
    print(args.N, args.save_file, args.equilibrium)
    main(args)
