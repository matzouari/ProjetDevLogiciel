"""
Le fichier où l'on regroupe toutes les méthodes pour produire l'application finalisée
"""

from AlgoGenetique import algo_genetique
from VAE import auto_bon
import torch
import torchvision.models as models

# Définis l'architecture de ton autoencodeur (encodeur et décodeur)
# Supposons que tu aies déjà défini ton autoencodeur comme 'Autoencoder'

# Charge les paramètres enregistrés
latent_dim = 6

checkpoint = torch.load("VAE/vae_model.pth")
autoencoder = auto_bon.VAE(latent_dim)

# Charge les paramètres dans ton modèle
autoencoder.load_state_dict(checkpoint['model_state_dict'])

# Maintenant, tu peux accéder aux paramètres de ton décodeur
decoder_parameters = autoencoder.decoder.parameters()

# Fais quelque chose avec les paramètres du décodeur, par exemple :
latent_coordinates = torch.tensor([[0.5, -0.3]])

generated_image = autoencoder.decoder(latent_coordinates)

print("Dimensions de l'image générée:", generated_image.shape)