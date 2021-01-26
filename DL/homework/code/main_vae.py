import argparse

import matplotlib.pyplot as plt
import torch
import torchvision

from trainer import Trainer
from vae import data
from vae import loss
from vae import model


def generate_images(name, vae, z, device):
    vae.eval().to(device)
    z = z.to(device)
    with torch.no_grad():
        generated_images = vae.decoder(z).view((64, 1, 28, 28))
    torchvision.utils.save_image(generated_images, name)


def main(data_folder, batch_size, architecture, epochs):
    train_set, test_set = data.load_datasets(data_folder)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False)

    # Build the model
    if architecture == "conv":
        latent_dim = (20, 1, 1)
        encoder = model.ConvEncoder()  # Already designed for 28, 28 images and 20, 4, 4 latent dim
        decoder = model.ConvDecoder()
        vae = model.GaussianVae(encoder, decoder)
    else:
        latent_dim = (20,)
        dim = train_set[0][0].numel()
        encoder = model.MLPEncoder([dim, 350, 2*latent_dim[0]])
        decoder = model.MLPDecoder([latent_dim[0], 350, dim])
        vae = model.GaussianVae(encoder, decoder)

    # Prepare training
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = loss.ElboLoss()
    trainer = Trainer(vae, optimizer, save_mode="never")
    print("Using device", device)

    # Q10 before training
    z = torch.randn((64, *latent_dim))  # Random latent variables
    generate_images(f"{architecture}_0.png", vae, z, device)

    epochs = sorted(epochs)
    losses = torch.zeros(2, epochs[-1])

    for epoch in epochs:
        # Train util epoch
        losses[:, :epoch] += trainer.train(epoch, train_loader, criterion, device, val_loader=test_loader)
        # Save the image for Q10/Q11 after epoch of training
        generate_images(f"{architecture}_{epoch}.png", vae, z, device)

    plt.title("Evolution of the loss during training")
    plt.plot(losses[0], label="train")
    plt.plot(losses[1], label="val")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning Assignment: VAE")
    parser.add_argument(
        "epochs",
        type=int,
        nargs="+",
        help="Number of training epochs. If several passed, will produce an image for each of them.",
    )

    parser.add_argument(
        "-D",
        "--data-directory",
        default="data",
        help="Directory where the datasets are stored. Default to ./data",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        help="Batch size. Default to 64",
    )
    parser.add_argument(
        "-a",
        "--architecture",
        default="mlp",
        help="Architecture: [mlp|conv]. Default to mlp",
    )

    args = parser.parse_args()

    main(args.data_directory, args.batch_size, args.architecture.lower(), args.epochs)
