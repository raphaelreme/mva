import os

import torch

# TODO: use a logger rather than print (log where we want and not always check verbosity)
# TODO: Add scheduler ?
# TODO: Monitor time
# TODO: Load/save
# TODO: No longer use epochs but rather go through the loaders and eval every x steps
# TODO: Handle batch methods


class Trainer:
    """Own trainer class. Build for any classical pytorch project. Should wrap all the training procedures.

    Still unfinished. But work for what we need here.
    """

    def __init__(self, model, optimizer, verbose=True, save_mode="never", save_dir=".") -> None:
        """Constructor

        Args:
            model (nn.Module): The model to train
            optimizer (nn.Optimizer): The optimizer to use
            verbose (bool): Print meta info if True
            save_mode ("never"|"always"|"best"): When to save the model
            save_dir (str): Experiment directory, where models will be saved
        """
        self.model = model
        self.optimizer = optimizer
        self.verbose = verbose
        self.save_mode = save_mode
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)
        self.epoch = 0
        self.best_val = torch.tensor(float("inf"))
        self.best_epoch = -1
        self.losses = torch.zeros((2, 0))

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
        if n_epochs <= self.epoch:
            return self

        self.model.to(device)

        if val_criterion is None:
            val_criterion = criterion

        losses = torch.zeros((2, n_epochs))
        losses[:, :self.losses.shape[1]] = self.losses[:, :self.epoch]
        self.losses = losses

        while self.epoch < n_epochs:
            self.model.train()
            epoch_loss = 0.0
            N = 0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                batch_size = inputs.shape[0]
                N += batch_size

                predictions = self.model(inputs)
                loss = criterion(predictions, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss = loss.item()
                epoch_loss += loss * batch_size

                if self.verbose:
                    print(f"\rBatch {i+1}/{len(train_loader)} --- Training loss {loss:.4f}", end="", flush=True)
            if self.verbose:
                print(flush=True)

            if val_loader is None:
                val_loss = float("inf")
            else:
                val_loss = self.validate(val_loader, val_criterion, device)

            self.losses[0, self.epoch] = epoch_loss / N
            self.losses[1, self.epoch] = val_loss

            if self.verbose:
                print(
                    f"Epoch [{self.epoch}/{n_epochs}] --- Averaged training loss: {epoch_loss / N:.4f}"
                    f", Averaged validation loss: {val_loss:.4f}", flush=True)  # Timing ? # never reached n_epochs/n_epochs...

            if val_loss < self.best_val:
                self.best_val = val_loss
                self.best_epoch = self.epoch
                if self.save_mode == "best":
                    self.save()
            if self.save_mode == "always":
                self.save()

            self.epoch += 1

        return self

    def validate(self, dataloader, criterion, device):
        self.model.to(device)

        self.model.eval()
        with torch.no_grad():
            loss = 0
            N = 0

            for i, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                batch_size = inputs.shape[0]
                N += batch_size

                predictions = self.model(inputs)
                loss += criterion(predictions, targets).item() * batch_size

                if self.verbose:
                    print(f"\rBatch {i+1}/{len(dataloader)} --- Validation loss {loss / N:.4f}", end="", flush=True)
            if self.verbose:
                print(flush=True)

        return loss / N

    def save(self):
        # Should save also the optimizer !
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"{self.epoch}.pt"))

    def load(self, path):
        raise NotImplementedError
