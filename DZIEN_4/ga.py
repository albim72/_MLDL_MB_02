import random
import numpy as np
from deap import base, creator, tools


# Zdefiniuj funkcję oceny - minimalizujemy funkcję kwadratową
def eval_func(individual):
    return individual[0] ** 2,


# Utwórz typy obiektów i rejestrator
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Zdefiniuj toolbox
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -5.0, 5.0)  # Zakres wartości dla pojedynczej cechy
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)  # Inicjalizacja osobnika
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # Inicjalizacja populacji
toolbox.register("evaluate", eval_func)  # Funkcja oceny
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Operator krzyżowania
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)  # Operator mutacji
toolbox.register("select", tools.selTournament, tournsize=3)  # Operator selekcji


def main():
    random.seed(42)

    pop_size = 50
    num_generations = 100

    # Inicjalizacja populacji
    population = toolbox.population(n=pop_size)
    offspring = None

    # Ocena początkowej populacji
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    for gen in range(num_generations):
        print(f"Generation {gen + 1}")

        # Wybór rodziców
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Krzyżowanie
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

        # Mutacja
        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values

        # Ocena osobników zmutowanych
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Zamiana populacji na potomstwo
        population[:] = offspring

        # Statystyki
        fits = [ind.fitness.values[0] for ind in population]

        print(f"  Min: {min(fits)}")
        print(f"  Max: {max(fits)}")
        print(f"  Avg: {np.mean(fits)}")
        print(f"  Std: {np.std(fits)}")

    print("\n-- End of (successful) evolution --")

    best_ind = tools.selBest(population, 1)[0]
    print("Best individual:", best_ind)
    print("Best fitness:", best_ind.fitness.values[0])


if __name__ == "__main__":
    main()
