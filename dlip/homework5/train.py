import argparse
import numpy as np
import torch

from data import TrajectoryDataset
from model import LinearModel
from rossler_map import RosslerMap


def train(model, optimizer, criterion, device, dataloader, n_iter=1):
    """One epoch training"""
    model.train()
    cum_loss = 0
    for i, trajs in enumerate(dataloader):
        trajs = trajs.to(device)
        x = trajs  # batch, seq_len, dim

        losses = []
        for j in range(1, n_iter + 1):  # Compute recursively n_iter sequence and the associated loss
            x = model(x)
            losses.append(criterion(x[:, :-j, :], trajs[:, j:, :]))

        optimizer.zero_grad()
        loss = sum(losses) / len(losses)
        loss.backward()
        optimizer.step()
        cum_loss += loss.item()
    return cum_loss

                
def evaluate(model, criterion, device, dataloader):
    model.eval()
    with torch.no_grad():
        losses = []
        for i, trajs in enumerate(dataloader):
            trajs = trajs.to(device)
            batch_size, seq_len, dim = trajs.shape
            trajs = trajs.permute(1, 0, 2)
            cum_loss = 0
            N = 0

            x = trajs[0]  # First elt for each batch
            for j in range(1, seq_len):  # Recursively apply the model
                x = model(x)
                cum_loss += criterion(x, trajs[j]).item()
                N += 1

            losses.append(cum_loss / N)
    return sum(losses) / len(losses)


def main(args):
    # Generate random data
    np.random.seed(42)
    rossler = RosslerMap(delta_t=0.1)
    train_dataset = TrajectoryDataset(rossler, args.n_traj, args.traj_duration, p_eq=args.equilibrium)
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True)

    # Validation: 20 trajectory of 100s.
    val_dataset = TrajectoryDataset(rossler, 20, 100, p_eq=0.5)
    val_loader = torch.utils.data.DataLoader(val_dataset, 20, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = LinearModel((3, 50, 100, 200, 3)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    best_val = float("inf")

    for i in range(args.epochs):
        train_loss = train(model, optimizer, criterion, device, train_loader, args.n_iter)
        val_loss = evaluate(model, criterion, device, val_loader)
        print(f"Epoch {i}/{args.epochs}: Train loss: {train_loss}, Val loss: {val_loss}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--equilibrium", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--traj-duration", type=int, default=200)
    parser.add_argument("--n-traj", type=int, default=100)
    parser.add_argument("--n-iter", type=int, default=1)
    parser.add_argument("--save-file", default="best.pth")
    args = parser.parse_args()
    main(args)
