from __future__ import annotations
import random
from typing import List, Callable, Tuple

from ga_schedule import (
    GAConfig,
    GAHistory,
    fitness,
    random_chromosome,
    tournament_selection,
    one_point_crossover,
    mutate,
)


# Альтернативные операторы селекции и кроссовера

def roulette_selection(population: List[List[int]], fitnesses: List[float]) -> List[int]:
    """
    Рулеточная селекция: вероятность выбора особи пропорциональна её приспособленности.
    """
    total_fit = sum(fitnesses)
    # на случай очень маленьких или нулевых фитнесов
    if total_fit == 0:
        return population[random.randrange(len(population))][:]

    pick = random.random() * total_fit
    current = 0.0
    for chrom, fit in zip(population, fitnesses):
        current += fit
        if current >= pick:
            return chrom[:]  # копия хромосомы
    return population[-1][:]


def two_point_crossover(
    parent1: List[int],
    parent2: List[int],
    crossover_prob: float,
) -> Tuple[List[int], List[int]]:
    """
    Двухточечный кроссовер: выбираются две точки, средние сегменты родителей обмениваются.
    """
    if random.random() > crossover_prob or len(parent1) <= 2:
        return parent1[:], parent2[:]

    n = len(parent1)
    p1, p2 = sorted(random.sample(range(1, n), 2))
    child1 = parent1[:p1] + parent2[p1:p2] + parent1[p2:]
    child2 = parent2[:p1] + parent1[p1:p2] + parent2[p2:]
    return child1, child2


# Запуск ГА с разными стратегиями

def run_ga_custom(
    lessons,
    config: GAConfig | None = None,
    selection: str = "tournament",
    crossover: str = "one_point",
) -> GAHistory:
    """
    Запуск генетического алгоритма с возможностью выбора стратегии селекции и кроссовера.
    Возвращает объект GAHistory, совместимый с основной реализацией.
    """
    if config is None:
        config = GAConfig()

    num_lessons = len(lessons)

    # выбор функции селекции
    if selection == "tournament":
        def select(pop, fits):
            return tournament_selection(pop, fits, config.tournament_size)
    elif selection == "roulette":
        def select(pop, fits):
            return roulette_selection(pop, fits)
    else:
        raise ValueError(f"Unknown selection strategy: {selection}")

    # выбор кроссовера
    if crossover == "one_point":
        cross_fn: Callable[[List[int], List[int], float], Tuple[List[int], List[int]]] = one_point_crossover
    elif crossover == "two_point":
        cross_fn = two_point_crossover
    else:
        raise ValueError(f"Unknown crossover type: {crossover}")

    # инициализация популяции
    population: List[List[int]] = [
        random_chromosome(num_lessons) for _ in range(config.population_size)
    ]

    best_fitness_per_gen: List[float] = []
    avg_fitness_per_gen: List[float] = []
    global_best_chrom: List[int] = []
    global_best_fit: float = -1.0

    for _ in range(config.generations):
        fitnesses = [fitness(chrom, lessons) for chrom in population]

        best_fit = max(fitnesses)
        avg_fit = sum(fitnesses) / len(fitnesses)

        best_fitness_per_gen.append(best_fit)
        avg_fitness_per_gen.append(avg_fit)

        if best_fit > global_best_fit:
            global_best_fit = best_fit
            global_best_chrom = population[fitnesses.index(best_fit)][:]

        # формируем новое поколение (с элитой)
        new_population: List[List[int]] = []
        new_population.append(global_best_chrom[:])  # элита

        while len(new_population) < config.population_size:
            parent1 = select(population, fitnesses)
            parent2 = select(population, fitnesses)

            child1, child2 = cross_fn(parent1, parent2, config.crossover_prob)

            mutate(child1, config.mutation_prob)
            mutate(child2, config.mutation_prob)

            new_population.append(child1)
            if len(new_population) < config.population_size:
                new_population.append(child2)

        population = new_population[:config.population_size]

    return GAHistory(
        best_fitness_per_gen=best_fitness_per_gen,
        avg_fitness_per_gen=avg_fitness_per_gen,
        best_chromosome=global_best_chrom,
        best_fitness=global_best_fit,
    )
