import random
import math


# Fonction pour initialiser une population de vecteurs avec un bruit à partir d'un vecteur initial
def create_new_photos(nombre_photos, base_vector, noise_factor):
    """
    """
    coordonnees_photos = []
    for _ in range(nombre_photos):
        new_vector = [coord + random.uniform(-noise_factor, noise_factor) for coord in base_vector]
        coordonnees_photos.append(new_vector)
    return coordonnees_photos

# Methode 1 : calcule le vecteur de coordonnées des centroides des vecteurs fournis puis génère une population de vecteurs
def photos_methode_centroide(nombre_photos, vectors, noise_factor=3):
    """
    """
    if not vectors:
        return None  # Retourner None si la liste de vecteurs est vide
    dimensions = len(vectors[0])  # Nombre de dimensions
    centroid_vector = [0] * dimensions  # Initialiser le vecteur centroïde à zéro
    num_vectors = len(vectors)  # Nombre de vecteurs
    for vector in vectors:
        for i in range(dimensions):
            centroid_vector[i] += vector[i]  # Ajouter les coordonnées de chaque vecteur
    centroid_vector = [coord / num_vectors for coord in centroid_vector]  # Calculer la moyenne

    coords_photos = create_new_photos(nombre_photos, centroid_vector, noise_factor)
    return coords_photos

# Methode 2 : crée un nouveau vecteur composé des coordonnées de tous les vecteurs de manière aléatoire puis génère une population de vecteurs
def photos_methode_crossover(nombre_photos, vectors, noise_factor = 3):
    """
    """
    new_vector = []
    for i in range(len(vectors[0])):
        coord = random.randint(0, len(vectors)-1)
        new_vector.append(vectors[coord][i])

    coords_photos = create_new_photos(nombre_photos, new_vector, noise_factor)
    return coords_photos

# Methode 3 : applique le bruit sur chacun des vecteurs avant le regroupement
def photos_methode_noise(nombre_photos, vectors):
    """
    """
    

if __name__ == "__main__":
    vector_1 = [10,20,30,40,50,60]
    vector_2 = [7,73,42,901,52,6]
    vector_3 = [17,74,63,716,893,42]
    vector_4 = [643,27,10,4,746,123]
    vector_list = [vector_1,vector_2,vector_3,vector_4]

    vector_method1 = photos_methode_centroide(10, vector_list)
    vector_method2 = photos_methode_crossover(10, vector_list)

    print(vector_method1)
    print(vector_method2)
