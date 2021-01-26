import torch


def train(model, train_loader, optimizer, criterion, epoch, device, logging_freq=10):
    model.train()
    for batch_idx, (data, _, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        batch, seq_len, out_shape = target.shape

        optimizer.zero_grad()
        output, _ = model(data)

        loss = criterion(output, target) / (batch * seq_len * out_shape)
        loss.backward()
        optimizer.step()

        if batch_idx % logging_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item(),
            ))


def validation(model, val_loader, criterion, device):
    model.eval()
    loss = 0
    size = 0
    for data, _, target in val_loader:
        data, target = data.to(device), target.to(device)
        batch, seq_len, out_shape = target.shape
        size += batch * seq_len * out_shape

        with torch.no_grad():
            output, _ = model(data)
            loss += criterion(output, target).data.item()

    loss /= size
    print('\nValidation set: Average loss: {:.4f}'.format(loss))
    return loss
