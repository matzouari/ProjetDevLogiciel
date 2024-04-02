"""
Le fichier où l'on regroupe toutes les méthodes pour produire l'application finalisée
"""

from AlgoGenetique import algo_genetique
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import random

class VAE(nn.Module):
    """
    Implémente un Variational Autoencoder (VAE) pour la génération d'images.

    Le VAE est composé d'un encodeur et d'un décodeur, qui apprennent à
    représenter et à générer des données, respectivement.

    Args:
        latent_dim (int): La dimension de l'espace latent.

    Attributes:
        encoder (nn.Sequential): Le réseau de neurones de l'encodeur.
        decoder (nn.Sequential): Le réseau de neurones du décodeur.
    """

    def __init__(self, latent_dim: int):
        """
        Initialise une instance de VAE avec la dimension de l'espace latent spécifiée.

        Args:
            latent_dim (int): La dimension de l'espace latent.
        """
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor, variance_scale=0.1) -> torch.Tensor:
        """
        Effectue la reparamétrisation nécessaire pour échantillonner dans l'espace latent.

        Args:
            mu (torch.Tensor): La moyenne de la distribution latente.
            log_var (torch.Tensor): Le logarithme de la variance de la distribution latente.

        Returns:
            torch.Tensor: L'échantillon dans l'espace latent.
        """
        log_var = log_var*variance_scale
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passe l'entrée à travers l'encodeur et le décodeur pour la reconstruction.

        Args:
            x (torch.Tensor): L'image d'entrée.

        Returns:
            torch.Tensor: L'image reconstruite, la moyenne de la distribution latente
                et le logarithme de la variance de la distribution latente.
        """
        z_params = self.encoder(x)
        mu, log_var = torch.chunk(z_params, 2, dim=-1)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

# Charge les paramètres enregistrés
latent_dim = 64

autoencoder = VAE(latent_dim)

checkpoint = torch.load("CriminAI/CriminAI/models/celebA_mix_MSE_lossL1.pth")
autoencoder.load_state_dict(checkpoint)

# Maintenant, tu peux accéder aux paramètres de ton décodeur
decoder_parameters = autoencoder.decoder.parameters()

# Fais quelque chose avec les paramètres du décodeur, par exemple :
image_coords = [random.uniform(-1, 1) for _ in range(latent_dim)]
image_coords2 = [random.uniform(-1, 1) for _ in range(latent_dim)]
latent_coordinates = torch.tensor([image_coords])
latent_coordinates2 = torch.tensor([image_coords2])

generated_image = autoencoder.decoder(latent_coordinates)
generated_image2 = autoencoder.decoder(latent_coordinates2)
image = generated_image.squeeze().detach().numpy()
image2 = generated_image2.squeeze().detach().numpy()

# Transpose les dimensions pour que les canaux soient le dernier axe
image = np.clip(image.transpose(1, 2, 0), 0, 1)
image2 = np.clip(image2.transpose(1, 2, 0), 0, 1)

print("Dimensions de l'image générée:", image.shape)
plt.imshow(image)
plt.show()

print("Dimensions de l'image générée:", image2.shape)
plt.imshow(image2)
plt.show()

new_image_coords = algo_genetique.photos_methode_crossover(4,[image_coords,image_coords2], 2)
print(new_image_coords)
new_images = []
n = len(new_image_coords)

for i in range(n):
    new_latent_coords = torch.tensor(new_image_coords[i])
    new_gen_image = autoencoder.decoder(new_latent_coords)
    new_images.append(new_gen_image.squeeze().detach().numpy())
    print("Dimensions de l'image générée:", new_gen_image.shape)

for i in range(n):
    # Afficher les images reconstruites
    ax = plt.subplot(1, n, i + 1)
    new_image = np.clip(new_images[i].transpose(1, 2, 0), 0, 1)
    plt.imshow(new_image)
    plt.title('Image Reconstruite')
    plt.axis('off')
plt.show()
