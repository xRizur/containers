import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Definicje typów kontenerów
CORRIDOR, RESIDENTIAL, KITCHEN, SANITARY, COMMON = range(5)
CONTAINER_TYPES = [CORRIDOR, RESIDENTIAL, KITCHEN, SANITARY, COMMON]

# Wymiary siatki
GRID_WIDTH, GRID_HEIGHT = 10, 10

# Kolory i etykiety dla wizualizacji
colors = ['grey', 'blue', 'green', 'red', 'yellow']
labels = ['Korytarz', 'Mieszkalny', 'Kuchnia', 'Sanitariaty', 'Przestrzeń wspólna']
cmap = mcolors.ListedColormap(colors)


def fitnessFunction(individual):
    grid = np.array(individual).reshape(GRID_HEIGHT, GRID_WIDTH)

    # Liczba kontenerów różnego typu
    num_residential = np.sum(grid == RESIDENTIAL)
    num_kitchen = np.sum(grid == KITCHEN)
    num_common = np.sum(grid == COMMON)
    num_sanitary = np.sum(grid == SANITARY)

    # Oblicz średnią odległość od kontenerów mieszkalnych do innych typów kontenerów
    kitchen_distance, sanitary_distance, common_distance, = calculateAverageDistance(grid)

    # Optymalne proporcje
    ideal_residential_per_kitchen = 5
    ideal_residential_per_common = 10
    ideal_residential_per_sanitary = 3

    # Obliczenie kar za nieoptymalne proporcje
    kitchen_score = abs(num_residential - ideal_residential_per_kitchen * num_kitchen) * 5
    common_score = abs(num_residential - ideal_residential_per_common * num_common) * 5
    sanitary_score = abs(num_residential - ideal_residential_per_sanitary * num_sanitary) * 5

    # Ocena dostępności do korytarza
    accessibility_score = evaluateAccessibility(grid)

    residential_score = num_residential * (-20)

    corridor_score = isCorridorFullyConnectedScore(grid) * 15

    # Łączna ocena
    total_score = kitchen_score + common_score + sanitary_score + accessibility_score + \
                  kitchen_distance * 18 + sanitary_distance * 30 + common_distance * 9 + residential_score + corridor_score
    return total_score,


def calculateAverageDistance(grid):
    # def average_distance(locations1, locations2):
    #     total_distance = 0
    #     num_pairs = 0
    #     ## Szukamy dystansu do najbliższego kontenera
    #     for loc1 in locations1:
    #         min_distance = 100
    #         for loc2 in locations2:
    #             distance = manhattanDistance(loc1, loc2)
    #             min_distance = min(min_distance, distance)
    #         total_distance += min_distance
    #         num_pairs += 1
    #
    #     return total_distance / num_pairs if num_pairs > 0 else 500
    def average_distance(locations1, locations2):
        total_distance = sum(manhattanDistance(loc1, loc2) for loc1 in locations1 for loc2 in locations2)
        return total_distance / (len(locations1) * len(locations2)) if locations1 and locations2 else 500

    residential_locations = [(i, j) for i in range(GRID_HEIGHT) for j in range(GRID_WIDTH) if grid[i][j] == RESIDENTIAL]
    kitchen_locations = [(i, j) for i in range(GRID_HEIGHT) for j in range(GRID_WIDTH) if grid[i][j] == KITCHEN]
    sanitary_locations = [(i, j) for i in range(GRID_HEIGHT) for j in range(GRID_WIDTH) if grid[i][j] == SANITARY]
    common_locations = [(i, j) for i in range(GRID_HEIGHT) for j in range(GRID_WIDTH) if grid[i][j] == COMMON]

    kitchen_average_distance = average_distance(residential_locations, kitchen_locations)
    sanitary_average_distance = average_distance(residential_locations, sanitary_locations)
    common_average_distance = average_distance(residential_locations, common_locations)

    return kitchen_average_distance, sanitary_average_distance, common_average_distance


def runCorridor(grid, true_matrix, i ,j):
    true_matrix[i][j] = 1
    if (j + 1 < len(grid[i]) and grid[i][j + 1] == CORRIDOR and true_matrix[i][j+1] == 0):
        runCorridor(grid, true_matrix, i, j+1)

    if(i + 1 < len(grid) and grid[i + 1][j] == CORRIDOR and true_matrix[i+1][j] == 0):
        runCorridor(grid, true_matrix, i+1, j)

    if (j - 1 >= 0 and grid[i][j - 1] == CORRIDOR and true_matrix[i][j-1] == 0):
        runCorridor(grid, true_matrix, i, j-1)

    if (i - 1 >= 0 and grid[i - 1][j] == CORRIDOR and true_matrix[i-1][j] == 0):
        runCorridor(grid, true_matrix, i-1, j)


def isCorridorFullyConnectedScore(grid):
    number_of_corridors = 0
    true_matrix = np.zeros((GRID_WIDTH,GRID_HEIGHT))
    for i in range(GRID_WIDTH):
        for j in range(GRID_HEIGHT):
            if grid[i][j] == CORRIDOR and true_matrix[i][j] == 0:
                number_of_corridors += 1
                runCorridor(grid, true_matrix, i ,j)
            else:
                true_matrix[i][j] = 1

    return number_of_corridors

def manhattanDistance(loc1, loc2):
    return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])


def isDirectlyConnectedToCorridor(grid, i, j):
    # Sprawdza, czy dana komórka jest bezpośrednio połączona z korytarzem
    return (j + 1 < len(grid[i]) and grid[i][j + 1] == CORRIDOR) or (
            i + 1 < len(grid) and grid[i + 1][j] == CORRIDOR) or (
            j - 1 > 0 and grid[i][j - 1] == CORRIDOR) or (
            i - 1 > 0 and grid[i - 1][j] == CORRIDOR)


def isCorridorConnectedToCorridor(grid, i, j):
    # Sprawdza, czy dana komórka korytarza jest połączona z więcej niż jednym korytarzem lub jest na brzegu gridu
    numberOfNeighbourCorridors = (int(j + 1 < len(grid[i]) and grid[i][j + 1] == CORRIDOR) + int(
        i + 1 < len(grid) and grid[i + 1][j] == CORRIDOR) + int(
        j - 1 >= 0 and grid[i][j - 1] == CORRIDOR) + int(
        i - 1 >= 0 and grid[i - 1][j] == CORRIDOR))
    logic_value = (numberOfNeighbourCorridors > 1) or \
                  (((j + 1 > len(grid[i])) or (j - 1 < 0) or (i + 1 > len(grid)) or (i - 1 < 0))
                   and numberOfNeighbourCorridors > 0)
    return logic_value


def evaluateAccessibility(grid):
    accessibility_score = 0

    # Sprawdzenie, czy każdy kontener jest połączony z korytarzem
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            if grid[i][j] in [RESIDENTIAL, KITCHEN, COMMON, SANITARY]:
                if not isDirectlyConnectedToCorridor(grid, i, j):
                    accessibility_score += 30  # Duża kara za brak bezpośredniego połączenia
            if grid[i][j] == CORRIDOR:
                if not isCorridorConnectedToCorridor(grid, i, j):
                    accessibility_score += 100  # Kara za niepołączone ze sobą korytarze

    return accessibility_score


# Inicjalizacja DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_item", lambda: CORRIDOR)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, GRID_WIDTH * GRID_HEIGHT)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitnessFunction)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(CONTAINER_TYPES) - 1, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    # Inicjalizacja populacji
    pop = toolbox.population(n=100)

    # Inicjalizacja Hall of Fame (najlepsze znalezione rozwiązanie)
    hof = tools.HallOfFame(1)

    # Statystyki
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Parametry algorytmu ewolucyjnego
    prob_cross = 0.7
    prob_mut = 0.5
    num_generations = 500

    # Uruchomienie algorytmu ewolucyjnego
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=prob_cross, mutpb=prob_mut,
                                   ngen=num_generations, stats=stats,
                                   halloffame=hof, verbose=True)

    # Wizualizacja najlepszego rozwiązania
    best = np.array(hof[0]).reshape(GRID_HEIGHT, GRID_WIDTH)
    # placeCorridor(best)  # Umieszczenie korytarza przed wizualizacją
    print("Najlepsze znalezione rozwiązanie:")
    print(best)
    plotGrid(best)

    return pop, log, hof


# Wizualizacja
def plotGrid(grid):
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(grid, cmap=cmap, aspect='auto')

    # Dodanie etykiet do siatki
    for (j, i), value in np.ndenumerate(grid):
        text = ''
        if value == RESIDENTIAL:
            text = 'M'  # Mieszkalny
        elif value == KITCHEN:
            text = 'K'  # Kuchnia
        elif value == SANITARY:
            text = 'S'  # Sanitariaty
        elif value == COMMON:
            text = 'W'  # Przestrzeń wspólna
        elif value == CORRIDOR:
            text = 'C'  # Korytarz
        plt.text(i, j, text, ha='center', va='center', color='black')

    plt.colorbar(cax, ticks=np.arange(len(colors)))
    plt.xticks(ticks=np.arange(GRID_WIDTH), labels=np.arange(GRID_WIDTH))
    plt.yticks(ticks=np.arange(GRID_HEIGHT), labels=np.arange(GRID_HEIGHT))
    plt.title("Rozmieszczenie kontenerów")
    plt.show()


if __name__ == "__main__":
    main()
