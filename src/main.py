"""
Le fichier où l'on regroupe toutes les méthodes pour produire l'application finalisée
"""

from AlgoGenetique import algo_genetique
from VAE import autoencodeur
import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import random

# Définis l'architecture de ton autoencodeur (encodeur et décodeur)
# Supposons que tu aies déjà défini ton autoencodeur comme 'Autoencoder'

# Charge les paramètres enregistrés
latent_dim = 64

checkpoint = torch.load("src/VAE/vae_model.pth")
autoencoder = autoencodeur.VAE(latent_dim)

# Charge les paramètres dans ton modèle
autoencoder.load_state_dict(checkpoint)

# Maintenant, tu peux accéder aux paramètres de ton décodeur
decoder_parameters = autoencoder.decoder.parameters()

# Fais quelque chose avec les paramètres du décodeur, par exemple :
image_coords = [random.uniform(-1, 1) for _ in range(latent_dim)]
print(image_coords)
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
