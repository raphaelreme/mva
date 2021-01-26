import torch
from torch import nn


class GaussianVae(nn.Module):
    """Variationnal AutoEncoder using Gaussian law:

    Use Gaussian laws in our latent space: q(z|x) = N(z; encode(x)_1, Diag(exp(encode(x)))_2)
    And p(z) = N(0, I)

    In order to enforce the we have a Positive Definite Matrix Sigma, we will use a log_trick.
    Assume that the encoder output the log of sigma and take the exp.

    Forward pass will return all what is needed to compute the loss:
    mu, log_sigma, and x_pred

    Attrs:
        encode (nn.Module): Encoder module, generate from x the parameters the gaussian law.
            Input (batch_size, dim) -> Output (batch_size, 2 * latent_dim)
        decode (nn.Module): Decoder module, reconstruct x from the latent space.
            Input (batch_size, latent_dim) -> Output (batch_size, dim)
    """
    def __init__(self, encoder_module, decoder_module):
        super().__init__()
        self.encoder = encoder_module
        self.decoder = decoder_module

    def forward(self, x):
        dim = x.shape

        mu, log_sigma = self.encoder(x)

        std = torch.exp(0.5 * log_sigma)  # Take exp + sqrt in order to have the std.

        # Reparametrization trick:
        eps = torch.randn_like(mu)
        z = eps * std + mu  # No need to create Diag(std)

        return mu, log_sigma, self.decoder(z).view(dim)  # Reshape if needed


class MLP(nn.Module):
    """MLP module for images

    Create simply a Sequential module with Linear layers and an activation between each linear.

    Will be use as decoder and encoder for our Vae module. (Outputs logits => Use BCEWithLogits rather than BCE in our loss)
    """
    def __init__(self, sizes, activation=nn.ReLU):
        super().__init__()

        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(activation())
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Question 9-10 MLP encoder/decoder
class MLPEncoder(MLP):
    def forward(self, x):
        x = super().forward(x.flatten(1))  # Flatten the image
        latent_dim = x.shape[-1] // 2
        return x[..., :latent_dim], x[..., latent_dim:]


class MLPDecoder(MLP):
    pass



# Question 11 Conv encoder/decoder

# Batch norm fully convolutional. Best
class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(  # From (1, 28, 28)
            nn.Conv2d(1, 64, 3, 1, 1), # (64, 28, 28)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (64, 14, 14)
            nn.Conv2d(64, 64*2, 3, 1, 0),  # (128, 12, 12)
            nn.BatchNorm2d(64*2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (128, 6, 6)
            nn.Conv2d(64*2, 64*4, 3, 1, 0), # (256, 4, 4)
            nn.BatchNorm2d(64*4),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (256, 2, 2)
            nn.Conv2d(64*4, 40, 2, 1, 0), # (40, 1, 1)
        )

    def forward(self, x):
        x = self.net(x)
        return  x[..., :20, :, :], x[..., 20:, :, :]


class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential( # From (20, 1, 1)
            nn.ConvTranspose2d(20, 64 * 4, 4, 2, 0), # (256, 4, 4)
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1),  # (128, 8, 8)
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64*2, 64, 4, 2, 2),  # (64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # (32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1), # (1, 28, 28)
        )

    def forward(self, z):
        return self.net(z)


# # Another example with Conv + Linear encoder and Linear + Conv decoder. (Works without batch norm)
# class ConvEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(  # From (1, 28, 28)
#             nn.Conv2d(1, 32, 3, 1, 1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),  # (32, 14, 14)
#             nn.Conv2d(32, 64, 3, 1, 1), 
#             nn.ReLU(),
#             nn.MaxPool2d(2),  # (64, 7, 7)
#             nn.Flatten(start_dim=1),
#             nn.Linear(64 * 7 * 7, 40),
#         )

#     def forward(self, x):
#         x = self.net(x)
#         return  x[..., :20], x[..., 20:]


# class ConvDecoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(20, 32 * 7 * 7),
#             nn.ReLU(),
#             View((32, 7, 7)),
#             nn.ConvTranspose2d(32, 16, 2, 2),
#             # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
#             nn.Conv2d(16, 16, 3, 1, 1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 8, 2, 2),
#             # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
#             nn.Conv2d(8, 1, 3, 1, 1),  # (1, 28, 28)
#         )

#     def forward(self, z):
#         return self.net(z)


# class View(nn.Module):
#     def __init__(self, shape):
#         super().__init__()
#         self.shape = shape

#     def forward(self, x):
#         x = x.view((-1, *self.shape))
#         return x
