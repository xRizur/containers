import random
from deap import base, creator, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Definicja kolorów dla różnych typów kontenerów
colors = ['white', 'blue', 'green', 'red', 'yellow']
labels = ['Pusty', 'Mieszkalny', 'Kuchnia', 'Sanitariaty', 'Przestrzeń wspólna']
cmap = mcolors.ListedColormap(colors)

def plotGrid(grid):
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap=cmap)
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ticks=np.arange(len(colors)), 
                 orientation='vertical').set_ticklabels(labels)
    plt.title("Rozmieszczenie kontenerów")
    plt.show()
# Typy kontenerów
EMPTY, RESIDENTIAL, KITCHEN, SANITARY, COMMON = 0, 1, 2, 3, 4
CONTAINER_TYPES = [EMPTY, RESIDENTIAL, KITCHEN, SANITARY, COMMON]

# Parametry siatki
GRID_SIZE = 10

# Funkcja oceny
def fitnessFunction(individual):
    grid = np.array(individual).reshape(GRID_SIZE, GRID_SIZE)

    # Kryteria oceny
    num_residential = np.sum(grid == RESIDENTIAL)
    num_kitchen = np.sum(grid == KITCHEN)
    # Dodaj tutaj inne kryteria

    # Optymalna ilość mieszkań na kuchnię (na przykład)
    ideal_residential_per_kitchen = 5
    kitchen_score = abs(num_residential - ideal_residential_per_kitchen * num_kitchen)

    # Obliczenie odległości między kontenerami (przykładowa metoda)
    distance_score = 0
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i][j] == RESIDENTIAL:
                # Oblicz odległość od najbliższego korytarza, kuchni itp.
                pass

    # Suma ocen - można dostosować wagi dla różnych kryteriów
    total_score = kitchen_score + distance_score
    return (total_score,)
# Inicjalizacja DEAP
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))  # Używamy -1.0, ponieważ dążymy do minimalizacji
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_item", random.choice, CONTAINER_TYPES)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, GRID_SIZE**2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitnessFunction)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(CONTAINER_TYPES)-1, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


# Funkcja do wizualizacji siatki
def plotGrid(grid):
    plt.imshow(grid, cmap="viridis")
    plt.colorbar()
    plt.show()

# Główna funkcja
def maindef():
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats, halloffame=hof, verbose=True)

    # Wizualizacja najlepszego rozwiązania
    best = np.array(hof[0]).reshape(GRID_SIZE, GRID_SIZE)
    plotGrid(best)

    return pop, log, hof

if __name__ == "__main__":
    maindef()
