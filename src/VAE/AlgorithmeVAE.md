# Algorithme de l'autoencodeur variationnel pour la génération d'images
L'utilisation de l'autoencodeur variationnel dans le processus de sélection de CriminAI permet d'obtenir des images réalistes et diversifiées tout en préservant les caractéristiques importantes des données d'entrée. Le principe de cet algorithme est le suivant :



## Données d'Entrée :
Les données d'entrée pour l'autoencodeur variationnel sont plus de 200 000 images de célébrités issues de la base de données CelebA. Chaque image est représentée par un vecteur numérique à n=64 dimensions.

## Codage et Décodage :
L'autoencodeur variationnel commence par encoder chaque image en un espace latent de dimension inférieure. Cela réduit la dimensionnalité des données tout en préservant les caractéristiques importantes de l'image. Ensuite, ces codes latents sont décodés pour reconstruire les images originales.

## Apprentissage :
L'autoencodeur variationnel est entraîné sur un ensemble de données comprenant les images sélectionnées par l'utilisateur. L'objectif de l'entraînement est de minimiser la différence entre les images originales et les images reconstruites, tout en régularisant la distribution des codes latents pour suivre une distribution normale.

## Génération de Nouvelles Images :
Une fois l'autoencodeur variationnel entraîné, il peut être utilisé pour générer de nouvelles images. Cela se fait en échantillonnant aléatoirement des points dans l'espace latent et en les décodant pour obtenir les images correspondantes. Ces nouvelles images sont des variations de celles présentes dans l'ensemble d'entraînement.

## Optimisation des Paramètres :
Les performances de l'autoencodeur variationnel peuvent être améliorées en ajustant divers paramètres tels que la dimension de l'espace latent, les poids de la fonction de perte et les hyperparamètres de l'optimiseur.

## Sortie :
L'autoencodeur variationnel fournit en sortie les images reconstruites ainsi que les codes latents correspondants. Ces informations peuvent être utilisées pour sélectionner les images les plus pertinentes pour le logiciel CriminAI.
