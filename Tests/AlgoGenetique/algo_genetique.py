import random
import math
import numpy as np

# Fonction pour appliquer du bruit à chaque vecteur
def add_noise(vector, noise_factor):
    """
    Ajoute du bruit à un vecteur en lui ajoutant une petite valeur aléatoire à chaque coordonnée.

    :param vector: Le vecteur d'entrée.
    :type vector: list[float]
    :param noise_factor: Le facteur de bruit, déterminant l'amplitude du bruit ajouté.
    :type noise_factor: float
    :return: Le vecteur avec du bruit ajouté.
    :rtype: list[float]
    """
    noisy_vector = [coord + np.random.normal(0, noise_factor) for coord in vector]
    return noisy_vector


# Fonction pour initialiser une population de vecteurs avec un bruit à partir d'un vecteur initial
def create_new_photos(nombre_photos, base_vector, noise_factor):
    """
    Initialise une population de vecteurs avec un bruit à partir d'un vecteur initial.

    :param nombre_photos: Le nombre de photos à générer.
    :type nombre_photos: int
    :param base_vector: Le vecteur de base à partir duquel générer les nouvelles photos.
    :type base_vector: list[float]
    :param noise_factor: Le facteur de bruit à appliquer.
    :type noise_factor: float
    :return: Une liste de vecteurs représentant les nouvelles photos.
    :rtype: list[list[float]]
    """
    coordonnees_photos = []
    for _ in range(nombre_photos):
        new_vector = [add_noise(base_vector, noise_factor)]
        coordonnees_photos.append(new_vector)
    return coordonnees_photos

# Methode 1 : calcule le vecteur de coordonnées des centroides des vecteurs fournis puis génère une population de vecteurs
def photos_methode_centroide(nombre_photos, vectors, noise_factor=1):
    """
    Calcule le vecteur de coordonnées des centroides des vecteurs fournis puis génère une population de vecteurs.

    :param nombre_photos: Le nombre de photos à générer.
    :type nombre_photos: int
    :param vectors: Les vecteurs à partir desquels calculer le centroïde.
    :type vectors: list[list[float]]
    :param noise_factor: Le facteur de bruit à appliquer.
    :type noise_factor: float
    :return: Une liste de vecteurs représentant les nouvelles photos.
    :rtype: list[list[float]]
    """
    if not vectors:
        return None  # Retourner None si la liste de vecteurs est vide
    dimensions = len(vectors[0])  # Nombre de dimensions du vecteur
    centroid_vector = [0] * dimensions  # Initialiser le vecteur centroïde à zéro
    num_vectors = len(vectors)  # Nombre de vecteurs dans la liste
    for vector in vectors:
        for i in range(dimensions):
            centroid_vector[i] += vector[i]  # Ajouter les coordonnées de chaque vecteur
    centroid_vector = [coord / num_vectors for coord in centroid_vector]  # Calculer la moyenne des coordonnées

    # Générer une population de nouveaux vecteurs avec du bruit à partir du vecteur centroïde
    coords_photos = create_new_photos(nombre_photos, centroid_vector, noise_factor)
    return coords_photos

# Methode 2 : crée un nouveau vecteur composé des coordonnées de tous les vecteurs de manière aléatoire puis génère une population de vecteurs
def photos_methode_crossover(nombre_photos, vectors, noise_factor=1):
    """
    Crée un nouveau vecteur composé des coordonnées de tous les vecteurs de manière aléatoire puis génère une population de vecteurs.

    :param nombre_photos: Le nombre de photos à générer.
    :type nombre_photos: int
    :param vectors: Les vecteurs à partir desquels créer le nouveau vecteur.
    :type vectors: list[list[float]]
    :param noise_factor: Le facteur de bruit à appliquer.
    :type noise_factor: float
    :return: Une liste de vecteurs représentant les nouvelles photos.
    :rtype: list[list[float]]
    """
    new_vector = []
    for i in range(len(vectors[0])):
        coord = random.randint(0, len(vectors)-1)  # Sélectionner un vecteur de manière aléatoire
        new_vector.append(vectors[coord][i])  # Ajouter les coordonnées du vecteur sélectionné au nouveau vecteur

    # Générer une population de nouveaux vecteurs avec du bruit à partir du vecteur nouvellement créé
    coords_photos = create_new_photos(nombre_photos, new_vector, noise_factor)
    return coords_photos

# Methode 3 : applique le bruit sur chacun des vecteurs avant le regroupement
def photos_methode_noise(nombre_photos, vectors, noise_factor=1):
    """
    Applique le bruit sur chacun des vecteurs avant le regroupement puis génère une population de vecteurs.

    :param nombre_photos: Le nombre de photos à générer.
    :type nombre_photos: int
    :param vectors: Les vecteurs sur lesquels appliquer le bruit.
    :type vectors: list[list[float]]
    :param noise_factor: Le facteur de bruit à appliquer.
    :type noise_factor: float
    :return: Une liste de vecteurs représentant les nouvelles photos.
    :rtype: list[list[float]]
    """
    # Appliquer du bruit à chaque vecteur dans la liste
    noisy_vectors = [add_noise(vector, noise_factor) for vector in vectors]
    
    # Générer une population de nouveaux vecteurs avec du bruit supplémentaire ajouté à chaque vecteur
    new_vectors_population = []
    for _ in range(nombre_photos):
        # Sélectionner un vecteur bruité de manière aléatoire dans la liste et ajouter un peu de bruit supplémentaire
        new_vector = [coord + np.random.normal(0, noise_factor) for coord in noisy_vectors[random.randint(0, len(noisy_vectors)-1)]]
        new_vectors_population.append(new_vector)
    
    return new_vectors_population

if __name__ == "__main__":
    # Vecteurs de base pour les méthodes
    vector_1 = [10,20,30,40,50,60]
    vector_2 = [7,73,42,901,52,6]
    vector_3 = [17,74,63,716,893,42]
    vector_4 = [643,27,10,4,746,123]
    vector_list = [vector_1, vector_2, vector_3, vector_4]

    # Utilisation des méthodes pour générer des populations de vecteurs avec différentes techniques
    vector_method1 = photos_methode_centroide(10, vector_list)
    vector_method2 = photos_methode_crossover(10, vector_list)
    vector_method3 = photos_methode_noise(10, vector_list)

    # Affichage des résultats
    print(vector_method1)
    print(vector_method2)
    print(vector_method3)
