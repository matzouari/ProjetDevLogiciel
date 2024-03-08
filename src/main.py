"""
Le fichier où l'on regroupe toutes les méthodes pour produire l'application finalisée
"""

from AlgoGenetique import algo_genetique
from VAE import auto_bon
import torch
import torchvision.models as models
import matplotlib.pyplot as plt

# Définis l'architecture de ton autoencodeur (encodeur et décodeur)
# Supposons que tu aies déjà défini ton autoencodeur comme 'Autoencoder'

# Charge les paramètres enregistrés
latent_dim = 6

checkpoint = torch.load("/Users/matis/Documents/ECOLE/4A/ProjetDevLogiciel/src/VAE/vae_model.pth")
autoencoder = auto_bon.VAE(latent_dim)

# Charge les paramètres dans ton modèle
autoencoder.load_state_dict(checkpoint)

# Maintenant, tu peux accéder aux paramètres de ton décodeur
decoder_parameters = autoencoder.decoder.parameters()

# Fais quelque chose avec les paramètres du décodeur, par exemple :
latent_coordinates = torch.tensor([[0.5, -0.3, 0.2, 0.7, 0.4, -0.8]])

generated_image = autoencoder.decoder(latent_coordinates)
image = generated_image.squeeze().detach().numpy()

print("Dimensions de l'image générée:", image.shape)
plt.imshow(image)
plt.show()