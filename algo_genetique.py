import random
import math

# Définition des constantes
DIMENSIONS = 6
POPULATION_SIZE = 100
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1
VECTEUR_INITIAL = [10,20,30,40,50,60]

# Fonction pour initialiser une population de vecteurs
def initialize_population(population_size, base_vector, noise_factor=0.1):
    population = []
    for _ in range(population_size):
        new_vector = [coord + random.uniform(-noise_factor, noise_factor) for coord in base_vector]
        population.append(new_vector)
    return population

# Fonction pour évaluer la distance entre deux vecteurs
def distance(vector1, vector2):
    sum_squared_diff = sum((v1 - v2) ** 2 for v1, v2 in zip(vector1, vector2))
    return math.sqrt(sum_squared_diff)

# Fonction d'évaluation
def evaluation_function(vector):
    target_vector = [0.5] * DIMENSIONS  # Vecteur cible (avec toutes les coordonnées égales à 0.5)
    return -distance(vector, target_vector)  # On veut maximiser la similarité avec le vecteur cible

# Modification de la fonction fitness pour utiliser la nouvelle fonction d'évaluation
def fitness(vector):
    return evaluation_function(vector)

# Fonction pour sélectionner des vecteurs en fonction de leur qualité
def selection(population):
    return sorted(population, key=fitness)[:POPULATION_SIZE]

# Fonction pour créer de nouvelles solutions en croisant des vecteurs
def crossover(parent1, parent2):
    midpoint = random.randint(0, DIMENSIONS - 1)
    child = parent1[:midpoint] + parent2[midpoint:]
    return child

# Fonction pour muter un vecteur
def mutate(vector):
    for i in range(DIMENSIONS):
        if random.random() < MUTATION_RATE:
            vector[i] = random.uniform(0, 1)
    return vector

# Algorithme génétique
def genetic_algorithm():
    population = initialize_population(POPULATION_SIZE, VECTEUR_INITIAL)

    for generation in range(NUM_GENERATIONS):
        population = selection(population)
        new_population = []

        while len(new_population) < POPULATION_SIZE:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    # Sélectionner les meilleurs vecteurs
    best_vectors = selection(population)
    return best_vectors[:4]

# Exécuter l'algorithme génétique
best_vectors = genetic_algorithm()
for vector in best_vectors:
    print(vector)
