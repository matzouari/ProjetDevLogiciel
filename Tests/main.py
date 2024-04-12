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
from AlgoGenetique import algo_genetique
from VAE.autoencodeur_celebA_combinaison_loss_fc import VAE

# Charge les paramètres enregistrés
latent_dim = 500

autoencoder = VAE(latent_dim)

checkpoint = torch.load("allModels/model_celebA_mix_MSE_lossL1_3.pth")
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
