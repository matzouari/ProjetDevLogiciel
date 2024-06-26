import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import matplotlib.pyplot as plt



# Charger la base de données Olivetti Faces
data = fetch_olivetti_faces()
images = data.images
images = images.reshape(images.shape[0], 1, 64, 64)  # Adapter les images à la forme (channels, height, width)

# Convertir les données en tenseurs PyTorch
tensor_x = torch.tensor(images, dtype=torch.float32)

# Définir un DataLoader pour la gestion des données
batch_size = 64
data_loader = DataLoader(tensor_x, batch_size=batch_size, shuffle=True)

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


if __name__ == "__main__":
    # Définir les dimensions
    latent_dim = 64

    # Initialiser le modèle
    model = VAE(latent_dim)

    # Définir la fonction de perte et l'optimiseur
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Entraînement du modèle
    num_epochs = 200
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in data_loader:
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(batch)
            reconstruction_loss = criterion(recon_batch, batch)
            kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = 50*reconstruction_loss + kl_divergence #pondération à tester
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(data_loader.dataset):.4f}")

    # Sauvegarder le modèle
    torch.save(model.state_dict(), 'vae_model.pth')


    # Charger le modèle sauvegardé
    model = VAE(latent_dim)
    model.load_state_dict(torch.load('vae_model.pth'))
    model.eval()  # Mettre le modèle en mode évaluation

    # Prendre un batch d'images d'entrée
    with torch.no_grad():
        input_batch = next(iter(data_loader))

    # Reconstruire les images à partir du modèle
    recon_batch, _, _ = model(input_batch)

    # Convertir les tenseurs PyTorch en numpy arrays
    input_batch = input_batch.numpy()
    recon_batch = recon_batch.detach().numpy()

    # Afficher les images d'entrée et les images reconstruites
    n = 10  # Nombre d'images à afficher
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Afficher les images d'entrée
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(input_batch[i].reshape(64, 64), cmap='gray')
        plt.title('Image Originale')
        plt.axis('off')
        # Afficher les images reconstruites
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(recon_batch[i].reshape(64, 64), cmap='gray')
        plt.title('Image Reconstruite')
        plt.axis('off')
    plt.show()
