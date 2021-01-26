import os

import torch
from tqdm import tqdm


class Trainer:
    """Own trainer class. Build for any classical pytorch project. Should wrap all the training procedures.

    Still unfinished. But work for what we need here.
    """
    def __init__(self, model, optimizer, verbose=True, save_mode="best", save_dir=".") -> None:
        self.model = model
        self.optimizer = optimizer
        self.verbose = verbose
        self.save_mode = save_mode
        self.save_dir = save_dir


        os.makedirs(self.save_dir, exist_ok=True)
        self.epoch = 0
        self.best_val = torch.tensor(float("inf"))
        self.best_epoch = -1

    def __call__(self, *args, **kwargs):
        return self.train(*args, **kwargs)

    def train(self, n_epochs, train_loader, criterion, device, val_loader=None, val_criterion=None):
        try:
            return self._train(n_epochs, train_loader, criterion, device, val_loader, val_criterion)
        except:
            if self.save_mode != "never":
                self.epoch += 0.5
                self.save()
                self.epoch -= 0.5
            raise

    def _train(self, n_epochs, train_loader, criterion, device, val_loader=None, val_criterion=None):
        self.model.to(device)

        if val_criterion is None:
            val_criterion = criterion
        losses = torch.zeros((2, n_epochs))

        save_mode = self.save_mode
        if not val_loader and save_mode == "best":  # In case we can't check best, always save
            save_mode = "always"

        while self.epoch < n_epochs:
            self.model.train()
            epoch_loss = 0.0
            N = 0
            for i, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)

                batch_size = data.shape[0]
                N += batch_size

                prediction = self.model(data)
                loss = criterion(prediction, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * batch_size

                if self.verbose:
                    print(f"\rBatch {i}/{len(train_loader)} --- Training loss {epoch_loss/N:.4f}", end="", flush=True)
            print(flush=True)

            losses[0, self.epoch] = epoch_loss / N
            losses[1, self.epoch] = self.validate(val_loader, val_criterion, device)

            if self.verbose:
                print(f"Epoch [{self.epoch}/{n_epochs}] --- Val_loss: {losses[1, self.epoch]:.4f}", flush=True)  # Timing ?

            if losses[1, self.epoch] < self.best_val:
                self.best_val = losses[1, self.epoch]
                self.best_epoch = self.epoch
                if self.save_mode == "best":
                    self.save()
            if save_mode == "always":
                self.save()

            self.epoch += 1

        return losses

    def validate(self, dataloader, criterion, device):
        self.model.to(device)
        if dataloader is None:
            return float("inf")
        if self.verbose:
            dataloader = tqdm(dataloader)
        self.model.eval()
        with torch.no_grad():
            loss = 0
            N = 0

            for data, target in dataloader:
                data = data.to(device)
                target = target.to(device)

                batch_size = data.shape[0]
                N += batch_size

                prediction = self.model(data)
                loss += criterion(prediction, target).item() * batch_size

        return loss / N

    def save(self):
        # Should save also the optimizer !
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"{self.epoch}.pt"))

    def load(self, path):
        raise NotImplementedError
