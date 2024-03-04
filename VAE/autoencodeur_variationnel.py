import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

latent_dim = 2 # Dimension de l'espace latent (voir comment résultats varient si on augmente le nbr de dimension)

#### DEFINIR impage_shape




### Encodeur
encoder_input = keras.Input(shape=(image_shape))      # crée une entrée pour le modèle, spécifiant la forme des données d'entrée à l'encodeur
                                                      # image_shape estr variable définie auparavant qui représente la forme des images d'entrée (ex : largeur, hauteur, canaux) pour une image RGB)
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_input):
                                                      # couche de convolution avec 32 filtres, une taille de noyau de 3x3, une fonction d'activation ReLU, un pas de 2 (pour réduire la dimensionnalité spatiale) et un padding "same" pour maintenir la taille de l'entrée
                                                      # couches de convolution sont utilisées dans l'encodeur pour extraire des caractéristiques importantes des données d'entrée (comme des traits saillants dans une image) et pour réduire la dimensionnalité des données avant de passer à l'espace latent
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)                               # aplati la sortie de la dernière couche de convolution en un vecteur unidimensionnel

z_mean = layers.Dense(latent_dim, name="z_mean")(x)   # couche dense qui prend le vecteur aplati comme entrée et produit la moyenne des valeurs de l'espace latent
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x) # autre couche dense qui prend le vecteur aplati comme entrée et produit le logarithme de la variance des valeurs de l'espace latent

def sampling(args):                                   # échantillonne des valeurs de l'espace latent en utilisant la formule de l'échantillonnage reparamétrisé pour générer des échantillons à partir de la distribution latente apprise
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))  # Dans un VAE, au lieu de directement échantillonner des valeurs aléatoires pour l'espace latent, on génère des échantillons à partir de la distribution normale paramétrée par la moyenne et l'écart-type prédits par le réseau
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, name="z")([z_mean, z_log_var]) # production de la couche Lambda pour appliquer la fonction d'échantillonnage à la moyenne et au logarithme de la variance des valeurs de l'espace latent, produisant ainsi les valeurs de l'espace latent

encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name="encoder") # modèle Keras qui prend l'entrée de l'encodeur et produit la moyenne, le logarithme de la variance et les valeurs de l'espace latent en sortie. Cela représente l'encodeur complet de l'autoencodeur variationnel





### Décodeur
latent_inputs = keras.Input(shape=(latent_dim,))         #  crée une entrée pour le modèle décodeur, spécifiant la forme des données d'entrée à partir de l'espace latent
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)  # définit une couche dense qui prend les valeurs de l'espace latent en entrée et les transforme en un vecteur de taille 7x7x64. Cela permet de projeter les valeurs latentes dans un espace qui peut être remodelé en une image de taille spécifique
x = layers.Reshape((7, 7, 64))(x)                        #  remodelle le vecteur de taille 7x7x64 en un tenseur 3D de taille (7, 7, 64). Cela prépare les données pour les couches de convolution transposées qui suivent, qui sont utilisées pour reconstruire l'image à partir de la représentation latente
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)  # couche de convolution transposée avec 64 filtres, une taille de noyau de 3x3, une fonction d'activation ReLU, un pas de 2 (pour augmenter la dimensionnalité spatiale) et un padding "same" pour maintenir la taille de l'entrée
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)  # dernière couche de convolution transposée avec 1 seul filtre (puisque nous reconstruisons une image en niveau de gris, mettre un 3 à la place si on veut RVB), une taille de noyau de 3x3 et une activation sigmoïde. Cette couche produit la sortie du décodeur, qui est une image reconstruite

decoder = keras.Model(latent_inputs, decoder_output, name="decoder") # crée un modèle Keras qui prend les valeurs de l'espace latent en entrée et produit l'image reconstruite en sortie. Cela représente le décodeur complet de l'autoencodeur variationnel





# VAE
vae_input = encoder_input  # définit l'entrée du modèle VAE en utilisant les données d'entrée originales, qui sont les images fournies à l'encodeur. Cela signifie que les images d'entrée sont propagées à travers l'encodeur pour obtenir les valeurs de l'espace latent
vae_output = decoder(z)  # définit la sortie du modèle VAE en utilisant les valeurs de l'espace latent z échantillonnées. Ces valeurs sont passées à travers le décodeur pour reconstruire les images

vae = keras.Model(vae_input, vae_output, name="vae") # crée un modèle Keras qui prend les données d'entrée (images) en entrée et produit les images reconstruites en sortie. Cela représente le modèle VAE complet qui prend en charge l'encodage des images en valeurs de l'espace latent, ainsi que le décodage des valeurs de l'espace latent en images reconstruites





# Fonction de perte (calcule la perte pour un autoencodeur variationnel (VAE))
def vae_loss(inputs, outputs, z_mean, z_log_var):
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)  # calcule la perte de reconstruction en utilisant la fonction de perte de la cross-entropie binaire. Cette fonction est utilisée car les images sont généralement binaires (0 ou 1 pour chaque pixel) ou normalisées entre 0 et 1
    reconstruction_loss *= image_shape[0] * image_shape[1] # taille de l'image / ajuste la perte de reconstruction en multipliant par la taille de l'image. Cela normalise la perte en fonction de la taille de l'image pour que la perte soit cohérente quelles que soient les dimensions de l'image
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = -0.5 * tf.reduce_sum(kl_loss, axis=-1)
    return tf.reduce_mean(reconstruction_loss + kl_loss) # calcule la perte totale en ajoutant la perte de reconstruction et la divergence KL. tf.reduce_mean calcule la moyenne de la perte sur tous les exemples d'entraînement





# Compilation et entraînement du modèle

# Compilation du modèle VAE en spécifiant l'optimiseur à utiliser (dans ce cas, Adam) et la fonction de perte
# La fonction de perte est définie en utilisant une fonction lambda qui prend les entrées et les sorties du modèle (x, y) ainsi que les valeurs moyennes et de log variance de l'espace latent (z_mean, z_log_var)
# La fonction de perte utilisée est celle définie précédemment, vae_loss, qui calcule à la fois la perte de reconstruction et la divergence KL
vae.compile(optimizer='adam', loss=lambda x, y: vae_loss(x, y, z_mean, z_log_var)) # perte utilisée lors de l'entraînement du modèle VAE pour guider la mise à jour des poids du réseau lors de la rétropropagation

#  entraîne le modèle VAE sur les données d'entraînement.
# Les paramètres sont : x_train : les données d'entraînement, x_train : les données d'entraînement utilisées à la fois comme entrée et sortie du modèle (puisque c'est un autoencodeur),
# epochs : le nombre d'époques d'entraînement, soit le nombre de fois que l'ensemble des données d'entraînement est parcouru, batch_size : le nombre d'échantillons utilisés pour calculer la perte à chaque itération de l'entraînement. Les mises à jour des poids du réseau sont effectuées après chaque lot.
# Pendant l'entraînement, le modèle VAE utilise la fonction de perte spécifiée lors de la compilation pour calculer le gradient et mettre à jour les poids du réseau à l'aide de l'optimiseur Adam
# Une fois l'entraînement terminé, le modèle VAE aura appris à encoder les données d'entrée en valeurs de l'espace latent et à décoder ces valeurs latentes en reconstructions des données d'entrée
vae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size)
