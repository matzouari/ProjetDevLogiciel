"""
Le but de ce programme est de tester différentes méthodes de regroupement de vecteurs
en un seul vecteur pour traiter le cas où l'utilisateur souhaite choisir plusieurs visages 
de référence, et l'objectif serait donc de regrouper les différentes caractéristiques afin
de donner le vecteur final à l'algorithme génétique pour qu'il puisse produire de nouvelles 
images plus proches des besoins de l'utilisateur
"""

import random

# Method 1
def centroid(vectors):
    if not vectors:
        return None  # Retourner None si la liste de vecteurs est vide
    dimensions = len(vectors[0])  # Nombre de dimensions
    centroid_vector = [0] * dimensions  # Initialiser le vecteur centroïde à zéro
    num_vectors = len(vectors)  # Nombre de vecteurs
    for vector in vectors:
        for i in range(dimensions):
            centroid_vector[i] += vector[i]  # Ajouter les coordonnées de chaque vecteur
    centroid_vector = [coord / num_vectors for coord in centroid_vector]  # Calculer la moyenne
    return centroid_vector

# Method 2 
def crossover(vectors):
    child = []
    for i in range(len(vectors[0])):
        coord = random.randint(0, len(vectors)-1)
        child.append(vectors[coord][i])
    return child


vector_1 = [10,20,30,40,50,60]
vector_2 = [7,73,42,901,52,6]
vector_3 = [17,74,63,716,893,42]
vector_4 = [643,27,10,4,746,123]
vector_list = [vector_1,vector_2,vector_3,vector_4]

print(centroid(vector_list))
print(crossover(vector_list))