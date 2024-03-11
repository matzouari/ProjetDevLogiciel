"""
Le fichier où l'on regroupe toutes les méthodes pour produire l'application finalisée
"""

from AlgoGenetique import algo_genetique
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import random

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
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
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        z_params = self.encoder(x)
        mu, log_var = torch.chunk(z_params, 2, dim=-1)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

# Charge les paramètres enregistrés
latent_dim = 64

checkpoint = torch.load("src/VAE/vae_model.pth")
autoencoder = VAE(latent_dim)

# Charge les paramètres dans ton modèle
autoencoder.load_state_dict(checkpoint)

# Maintenant, tu peux accéder aux paramètres de ton décodeur
decoder_parameters = autoencoder.decoder.parameters()

# Fais quelque chose avec les paramètres du décodeur, par exemple :
image_coords = [random.uniform(-1, 1) for _ in range(latent_dim)]
latent_coordinates = torch.tensor([image_coords])

generated_image = autoencoder.decoder(latent_coordinates)
image = generated_image.squeeze().detach().numpy()

print("Dimensions de l'image générée:", image.shape)
plt.imshow(image, cmap='gray')
plt.show()

new_image_coords = algo_genetique.photos_methode_crossover(4,[image_coords], 2)
new_images = []
n = len(new_image_coords)

for i in range(n):
    new_latent_coords = torch.tensor([new_image_coords[i]])
    new_gen_image = autoencoder.decoder(new_latent_coords)
    new_images.append(new_gen_image.squeeze().detach().numpy())

for i in range(n):
    # Afficher les images reconstruites
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(new_images[i], cmap='gray')
    plt.title('Image Reconstruite')
    plt.axis('off')
plt.show()