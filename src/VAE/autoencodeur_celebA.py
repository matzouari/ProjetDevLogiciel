import os
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


# Définir la variable d'environnement KMP_DUPLICATE_LIB_OK
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



class CustomDataset(Dataset):
    """
    Dataset personnalisé pour charger des images à partir d'un répertoire donné.

    Ce dataset charge les images à partir d'un répertoire spécifié et permet
    d'appliquer des transformations facultatives à ces images.

    Args:
        root_dir (str): Le chemin du répertoire contenant les images.
        transform (callable, optional): Une fonction/transform pour appliquer
            une transformation aux images. Default is None.

    Attributes:
        root_dir (str): Le chemin du répertoire contenant les images.
        transform (callable): La fonction/transform à appliquer aux images.
        image_list (list): La liste des noms de fichiers des images dans le répertoire.
    """

    def __init__(self, root_dir: str, transform=None):
        """
        Initialise une instance de CustomDataset avec les paramètres spécifiés.

        Args:
            root_dir (str): Le chemin du répertoire contenant les images.
            transform (callable, optional): Une fonction/transform pour appliquer
                une transformation aux images. Default is None.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = os.listdir(root_dir)

    def __len__(self) -> int:
        """
        Retourne le nombre total d'images dans le dataset.

        Returns:
            int: Le nombre total d'images dans le dataset.
        """
        return len(self.image_list)

    def __getitem__(self, idx: int):
        """
        Récupère une image du dataset à l'indice spécifié.

        Args:
            idx (int): L'indice de l'image à récupérer.

        Returns:
            PIL.Image.Image or torch.Tensor: L'image chargée du dataset.
        """
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

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

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Effectue la reparamétrisation nécessaire pour échantillonner dans l'espace latent.

        Args:
            mu (torch.Tensor): La moyenne de la distribution latente.
            log_var (torch.Tensor): Le logarithme de la variance de la distribution latente.

        Returns:
            torch.Tensor: L'échantillon dans l'espace latent.
        """
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



def train_VAE_model(model, data_loader, criterion, optimizer, num_epochs):
    """
    Entraîne le modèle VAE.

    Args:
        model (nn.Module): Le modèle VAE à entraîner.
        data_loader (DataLoader): Le DataLoader contenant les données d'entraînement.
        criterion: La fonction de perte utilisée pour l'entraînement.
        optimizer: L'optimiseur utilisé pour la mise à jour des poids du modèle.
        num_epochs (int): Le nombre d'époques pour lesquelles entraîner le modèle. Default is 100.

    Returns:
        trained_model (nn.Module): Le modèle VAE entraîné.
        reconstruction_losses (list): Liste des valeurs de perte de reconstruction pour chaque époque.
        kl_losses (list): Liste des valeurs de perte de divergence KL pour chaque époque.
        total_losses (list): Liste des valeurs de perte totale pour chaque époque.
    """
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

    return model, reconstruction_losses, kl_losses, total_losses



def plot_losses_and_parameters(model, reconstruction_losses, kl_losses, total_losses):
    """
    Affiche les paramètres du modèle et trace les courbes de perte.

    Args:
        model (nn.Module): Le modèle pour lequel afficher les paramètres.
        reconstruction_losses (list): Liste des valeurs de perte de reconstruction.
        kl_losses (list): Liste des valeurs de perte de divergence KL.
        total_losses (list): Liste des valeurs de perte totale.
    """
    # Afficher les paramètres du modèle
    for name, param in model.named_parameters():
        print(name, param.shape)

    # Tracer les courbes de perte
    plt.plot(reconstruction_losses, label='Reconstruction Loss')
    plt.plot(kl_losses, label='KL Loss')
    plt.plot(total_losses, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()


def visualize_images(model, data_loader):
    """
    Visualise les images originales et reconstruites après l'entraînement du modèle.

    Args:
        model (nn.Module): Le modèle VAE entraîné.
        data_loader (DataLoader): Le DataLoader contenant les données d'entraînement.
    """
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



if __name__ == "__main__":
    ### Chargement des données et autres configurations
    celeba_data_dir = "/Users/matis/Documents/ECOLE/4A/ProjetDevLogiciel/data/img_align_celeba" # Chemin vers le dossier contenant les images = Chemin où les données CelebA sont extraites
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
    latent_dim = 64 # Définir les dimensions de l'espace latent
    num_epochs=500 # Définir le nombre d'epochs


    ### Initialisation du modèle VAE, fonction de perte et optimiseur
    model = VAE(latent_dim) # Initialiser le modèle
    # Définir la fonction de perte et l'optimiseur
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    ### Entraînement du modèle VAE avec la fonction définie
    trained_model, reconstruction_losses, kl_losses, total_losses = train_VAE_model(model, data_loader, criterion, optimizer, num_epochs)


    ### Sauvegarder le modèle entraîné
    save_path = "/Users/matis/Documents/ECOLE/4A/ProjetDevLogiciel/data/vae_trained_model_celebA.pth"
    torch.save(trained_model.state_dict(), save_path)


    ### Charger le modèle sauvegardé
    model = VAE(latent_dim)
    model.load_state_dict(torch.load(save_path))
    model.eval()  # Mettre le modèle en mode évaluation


    ### Affichage des courbes de perte et des paramètres du modèle
    plot_losses_and_parameters(model, reconstruction_losses, kl_losses, total_losses)


    ### Visualisation des images reconstruites
    visualize_images(model, data_loader)
