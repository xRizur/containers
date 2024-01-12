import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Definicje typów kontenerów
EMPTY, RESIDENTIAL, KITCHEN, SANITARY, COMMON, CORRIDOR = range(6)
CONTAINER_TYPES = [EMPTY, RESIDENTIAL, KITCHEN, SANITARY, COMMON]

# Wymiary siatki
GRID_WIDTH, GRID_HEIGHT = 10, 10

# Kolory i etykiety dla wizualizacji
colors = ['white', 'blue', 'green', 'red', 'yellow', 'grey']
labels = ['Pusty', 'Mieszkalny', 'Kuchnia', 'Sanitariaty', 'Przestrzeń wspólna', 'Korytarz']
cmap = mcolors.ListedColormap(colors)

def fitnessFunction(individual):
    grid = np.array(individual).reshape(GRID_HEIGHT, GRID_WIDTH)
    
    # Umieszczenie korytarza
    placeCorridor(grid)

    # Liczba kontenerów różnego typu
    num_residential = np.sum(grid == RESIDENTIAL)
    num_kitchen = np.sum(grid == KITCHEN)
    num_common = np.sum(grid == COMMON)
    num_sanitary = np.sum(grid == SANITARY)
    
    # Oblicz średnią odległość od kontenerów mieszkalnych do innych typów kontenerów
    distance_score = calculateAverageDistance(grid)
    # Optymalne proporcje
    ideal_residential_per_kitchen = 5
    ideal_residential_per_common = 10
    ideal_residential_per_sanitary = 8

    # Obliczenie kar za nieoptymalne proporcje
    kitchen_score = abs(num_residential - ideal_residential_per_kitchen * num_kitchen)
    common_score = abs(num_residential - ideal_residential_per_common * num_common)
    sanitary_score = abs(num_residential - ideal_residential_per_sanitary * num_sanitary)

    # Ocena dostępności do korytarza
    accessibility_score = evaluateAccessibility(grid)

    # Łączna ocena
    total_score = kitchen_score + common_score + sanitary_score + accessibility_score + distance_score
    return total_score,

def calculateAverageDistance(grid):
    residential_locations = [(i, j) for i in range(GRID_HEIGHT) for j in range(GRID_WIDTH) if grid[i][j] == RESIDENTIAL]
    other_container_locations = [(i, j) for i in range(GRID_HEIGHT) for j in range(GRID_WIDTH) if grid[i][j] in [KITCHEN, SANITARY, COMMON]]

    total_distance = 0
    for res_loc in residential_locations:
        for other_loc in other_container_locations:
            total_distance += manhattanDistance(res_loc, other_loc)

    # Średnia odległość
    if residential_locations and other_container_locations:
        average_distance = total_distance / (len(residential_locations) * len(other_container_locations))
    else:
        average_distance = 0

    return average_distance

def manhattanDistance(loc1, loc2):
    return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
def placeCorridor(grid):
    # Tworzenie krzyżyka przez środek siatki
    middle_row = GRID_HEIGHT // 2
    middle_col = GRID_WIDTH // 2
    for i in range(GRID_HEIGHT):
        grid[i][middle_col] = CORRIDOR
    for j in range(GRID_WIDTH):
        grid[middle_row][j] = CORRIDOR

    # Znajdź kontenery na siatce
    container_positions = [(i, j) for i in range(GRID_HEIGHT) for j in range(GRID_WIDTH) if grid[i][j] in [RESIDENTIAL, KITCHEN, SANITARY, COMMON]]

    # Łącz kontenery z krzyżykiem, jeśli jeszcze nie są połączone
    for pos in container_positions:
        if not isDirectlyConnectedToCorridor(grid, pos[0], pos[1], middle_col):
            connectContainerWithCorridor(grid, pos[0], pos[1], middle_row, middle_col)

def isDirectlyConnectedToCorridor(grid, i, j, corridor_col):
    # Sprawdza, czy dana komórka jest bezpośrednio połączona z korytarzem
    return grid[i][corridor_col] == CORRIDOR or grid[GRID_HEIGHT // 2][j] == CORRIDOR

def connectContainerWithCorridor(grid, i, j, middle_row, middle_col):
    # Połącz kontener z najbliższym segmentem krzyżyka
    # Rysuj korytarz pionowo lub poziomo do środka, a potem do krzyżyka
    if abs(i - middle_row) < abs(j - middle_col):
        # Blizej środkowego rzędu, rysuj pionowo
        vertical_step = 1 if i < middle_row else -1
        while i != middle_row:
            grid[i][j] = CORRIDOR
            i += vertical_step
    else:
        # Blizej środkowej kolumny, rysuj poziomo
        horizontal_step = 1 if j < middle_col else -1
        while j != middle_col:
            grid[i][j] = CORRIDOR
            j += horizontal_step


def evaluateAccessibility(grid):
    accessibility_score = 0
    corridor_column = GRID_WIDTH // 2

    # Sprawdzenie, czy każdy kontener jest połączony z korytarzem
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            if grid[i][j] in [RESIDENTIAL, KITCHEN, COMMON, SANITARY]:
                if not isDirectlyConnectedToCorridor(grid, i, j, corridor_column):
                    accessibility_score += 10  # Duża kara za brak bezpośredniego połączenia

    # Dodatkowa kara za brak ciągłości korytarza
    if not isCorridorContinuous(grid, corridor_column):
        accessibility_score += 100

    return accessibility_score

def isDirectlyConnectedToCorridor(grid, i, j, corridor_col):
    # Sprawdza, czy dana komórka jest bezpośrednio połączona z korytarzem
    return grid[i, corridor_col] == CORRIDOR

def isCorridorContinuous(grid, corridor_col):
    # Sprawdza, czy korytarz jest ciągły
    return all(grid[i, corridor_col] == CORRIDOR for i in range(GRID_HEIGHT))

# ... (reszta kodu)


# Inicjalizacja DEAP
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_item", random.choice, CONTAINER_TYPES)
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
    prob_mut = 0.2
    num_generations = 400

    # Uruchomienie algorytmu ewolucyjnego
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=prob_cross, mutpb=prob_mut, 
                                   ngen=num_generations, stats=stats, 
                                   halloffame=hof, verbose=True)

    # Wizualizacja najlepszego rozwiązania
    best = np.array(hof[0]).reshape(GRID_HEIGHT, GRID_WIDTH)
    placeCorridor(best)  # Umieszczenie korytarza przed wizualizacją
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
