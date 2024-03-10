import os
# Définir la variable d'environnement KMP_DUPLICATE_LIB_OK
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CelebA
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


import wget
import zipfile
import PIL
from PIL import Image




class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

# Chemin vers le dossier contenant les images = Chemin où les données CelebA sont extraites
celeba_data_dir = "D:/img_align_celeba_short"

# Transformation des images
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Redimensionner les images à une taille de 64x64 pixels
    transforms.ToTensor(),  # Convertir les images en tenseurs PyTorch
])

# Charger les données à partir du dossier img_align_celeba
celeba_dataset = CustomDataset(root_dir=celeba_data_dir, transform=transform)

# Définir un DataLoader pour la gestion des données
batch_size = 64
data_loader = DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True)



class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # Modification ici : 3 canaux en entrée
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
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # Modification ici : 3 canaux en sortie
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



# Définir les dimensions
latent_dim = 64

# Initialiser le modèle
model = VAE(latent_dim)

# Définir la fonction de perte et l'optimiseur
criterion = nn.BCEWithLogitsLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Entraînement du modèle
num_epochs = 100
reconstruction_losses = []  # Liste pour stocker les valeurs de perte de reconstruction
kl_losses = []  # Liste pour stocker les valeurs de perte de divergence KL
total_losses = []  # Liste pour stocker les valeurs de perte totale
for epoch in range(num_epochs):
    total_loss = 0
    reconstruction_loss_total = 0
    kl_loss_total = 0
    for batch in data_loader:
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(batch)
        reconstruction_loss = criterion(recon_batch, batch)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = 50 * reconstruction_loss + kl_divergence  # Poids à tester
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        reconstruction_loss_total += reconstruction_loss.item()
        kl_loss_total += kl_divergence.item()

    # Calculer la perte moyenne par lot
    mean_reconstruction_loss = reconstruction_loss_total / len(data_loader.dataset)
    mean_kl_loss = kl_loss_total / len(data_loader.dataset)
    mean_total_loss = total_loss / len(data_loader.dataset)

    # Ajouter les valeurs de perte à la liste
    reconstruction_losses.append(mean_reconstruction_loss)
    kl_losses.append(mean_kl_loss)
    total_losses.append(mean_total_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(data_loader.dataset):.4f}")

# Tracer les courbes de perte
plt.plot(reconstruction_losses, label='Reconstruction Loss')
plt.plot(kl_losses, label='KL Loss')
plt.plot(total_losses, label='Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()


# Sauvegarder le modèle
# Définir le chemin complet du répertoire où vous souhaitez enregistrer le modèle
save_path = "/home/csutter/Documents/2023-2024/Développement  logiciel/vae_model_celebA.pth"
# Enregistrer le modèle
torch.save(model.state_dict(), save_path)


# Charger le modèle sauvegardé
model = VAE(latent_dim)
model.load_state_dict(torch.load(save_path))
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
    plt.imshow(np.clip(input_batch[i].transpose(1, 2, 0), 0, 1))  # Normaliser les valeurs entre 0 et 1
    plt.title('Image Originale')
    plt.axis('off')
    # Afficher les images reconstruites
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(np.clip(recon_batch[i].transpose(1, 2, 0), 0, 1))  # Normaliser les valeurs entre 0 et 1
    plt.title('Image Reconstruite')
    plt.axis('off')
plt.show()



#Voir ce qui ce passe dans le fichier vae_model.path
# Charger le modèle depuis le fichier vae_model.pth
model = VAE(latent_dim)
model.load_state_dict(torch.load(save_path))

# Afficher les paramètres du modèle
for name, param in model.named_parameters():
    print(name, param.shape)




## Tester avec plus grosse db (Celeb A)
## Courbe de décroissance de la perte (= courbe d'entrainement) : pour montrer que nos algo st bien performants
## Faire varier les hyperparamètres (nombre d’épochs, dimension de l’espace latent…)
## Amélioration de l'algo : faire varier les paramètres de variance de loi normale de l'espace latent

## Faire documentation sphinx sur autoencodeur variationnel
## Faire README sur autoencodeur variationnel
