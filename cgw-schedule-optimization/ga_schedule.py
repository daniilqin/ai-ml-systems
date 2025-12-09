from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Базовые константы расписания

DAYS = ["Пн", "Вт", "Ср", "Чт", "Пт"]
SLOTS_PER_DAY = 4
NUM_SLOTS = len(DAYS) * SLOTS_PER_DAY  # 20 слотов


# Модели данных

@dataclass
class Lesson:
    id: int
    name: str
    group: str
    teacher: str


@dataclass
class GAConfig:
    population_size: int = 50
    generations: int = 200
    crossover_prob: float = 0.8
    mutation_prob: float = 0.05
    tournament_size: int = 3


@dataclass
class GAHistory:
    best_fitness_per_gen: List[float]
    avg_fitness_per_gen: List[float]
    best_chromosome: List[int]
    best_fitness: float


# Пример учебной задачи расписания

def create_sample_lessons() -> List[Lesson]:
    """
    Создаёт фиксированный учебный пример:
    2 группы (G1, G2), 3 преподавателя (T1, T2, T3), 12 занятий.
    """
    lessons: List[Lesson] = []

    def add_lessons(name: str, group: str, teacher: str, count: int):
        nonlocal lessons
        for _ in range(count):
            lessons.append(
                Lesson(
                    id=len(lessons),
                    name=name,
                    group=group,
                    teacher=teacher
                )
            )

    # Группа G1
    add_lessons("Prog_G1", "G1", "T1", 3)
    add_lessons("Math_G1", "G1", "T2", 2)
    add_lessons("Econ_G1", "G1", "T3", 1)

    # Группа G2
    add_lessons("Prog_G2", "G2", "T1", 2)
    add_lessons("Math_G2", "G2", "T2", 3)
    add_lessons("Econ_G2", "G2", "T3", 1)

    return lessons


# Представление и декодирование хромосомы

def random_chromosome(num_lessons: int, num_slots: int = NUM_SLOTS) -> List[int]:
    """
    Хромосома: список длиной num_lessons.
    Значение гена = slot_id (0..num_slots-1).
    """
    return [random.randrange(num_slots) for _ in range(num_lessons)]


def decode_chromosome(chromosome: List[int], lessons: List[Lesson]) -> Dict[int, List[Lesson]]:
    """
    Превращает хромосому в расписание:
    slot_id -> список занятий в этом слоте.
    """
    schedule: Dict[int, List[Lesson]] = {s: [] for s in range(NUM_SLOTS)}
    for gene, slot_id in enumerate(chromosome):
        schedule[slot_id].append(lessons[gene])
    return schedule


# Подсчёт штрафов и fitness

def compute_penalty(
    chromosome: List[int],
    lessons: List[Lesson],
    num_slots: int = NUM_SLOTS,
    w_conflict: int = 10,
    w_window: int = 1,
    w_preference: int = 1,
) -> int:
    """
    Считаем суммарный штраф:
    - конфликты по преподавателям и группам;
    - окна в расписании групп;
    - нарушение простых предпочтений.
    """
    schedule = decode_chromosome(chromosome, lessons)
    penalty = 0

    # 1) Конфликты по преподавателям и группам
    for slot_id, slot_lessons in schedule.items():
        teachers: Dict[str, int] = {}
        groups: Dict[str, int] = {}
        for lesson in slot_lessons:
            teachers.setdefault(lesson.teacher, 0)
            teachers[lesson.teacher] += 1
            groups.setdefault(lesson.group, 0)
            groups[lesson.group] += 1
        # если один и тот же преподаватель/группа больше 1 раза -> конфликт
        for count in teachers.values():
            if count > 1:
                penalty += w_conflict * (count - 1)
        for count in groups.values():
            if count > 1:
                penalty += w_conflict * (count - 1)

    # 2) Окна по группам
    # собираем, какие слоты заняты для каждой группы
    group_slots: Dict[str, List[int]] = {}
    for gene, slot_id in enumerate(chromosome):
        group = lessons[gene].group
        group_slots.setdefault(group, []).append(slot_id)

    for group, slots in group_slots.items():
        # разбиваем по дням
        by_day: Dict[int, List[int]] = {}
        for slot_id in slots:
            day = slot_id // SLOTS_PER_DAY
            by_day.setdefault(day, []).append(slot_id)

        for day, day_slots in by_day.items():
            if len(day_slots) <= 1:
                continue
            day_slots_sorted = sorted(day_slots)
            first = day_slots_sorted[0]
            last = day_slots_sorted[-1]
            total_between = last - first + 1
            occupied = len(day_slots_sorted)
            windows = total_between - occupied
            if windows > 0:
                penalty += w_window * windows

    # 3) Простые предпочтения:
    # преподаватель T1 не любит первую пару (slot % 4 == 0),
    # группа G2 не любит последнюю пару (slot % 4 == 3)
    for gene, slot_id in enumerate(chromosome):
        lesson = lessons[gene]
        pair = slot_id % SLOTS_PER_DAY
        if lesson.teacher == "T1" and pair == 0:
            penalty += w_preference
        if lesson.group == "G2" and pair == 3:
            penalty += w_preference

    return penalty


def fitness(
    chromosome: List[int],
    lessons: List[Lesson],
    num_slots: int = NUM_SLOTS,
    **penalty_kwargs
) -> float:
    """
    Fitness = 1 / (1 + penalty)
    """
    p = compute_penalty(chromosome, lessons, num_slots=num_slots, **penalty_kwargs)
    return 1.0 / (1.0 + p)


# Операторы ГА: селекция, кроссовер, мутация

def tournament_selection(
    population: List[List[int]],
    fitnesses: List[float],
    tournament_size: int
) -> List[int]:
    indices = random.sample(range(len(population)), tournament_size)
    best_idx = max(indices, key=lambda i: fitnesses[i])
    # возвращаем копию
    return population[best_idx][:]


def one_point_crossover(
    parent1: List[int],
    parent2: List[int],
    crossover_prob: float
) -> Tuple[List[int], List[int]]:
    if random.random() > crossover_prob or len(parent1) <= 1:
        return parent1[:], parent2[:]
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(
    chromosome: List[int],
    mutation_prob: float,
    num_slots: int = NUM_SLOTS
) -> None:
    for i in range(len(chromosome)):
        if random.random() < mutation_prob:
            chromosome[i] = random.randrange(num_slots)


# Основной цикл ГА

def run_ga(
    lessons: List[Lesson],
    config: Optional[GAConfig] = None
) -> GAHistory:
    if config is None:
        config = GAConfig()

    num_lessons = len(lessons)

    # инициализация популяции
    population: List[List[int]] = [
        random_chromosome(num_lessons) for _ in range(config.population_size)
    ]

    best_fitness_per_gen: List[float] = []
    avg_fitness_per_gen: List[float] = []
    global_best_chrom: List[int] = []
    global_best_fit: float = -1.0

    for gen in range(config.generations):
        fitnesses = [fitness(chrom, lessons) for chrom in population]

        best_fit = max(fitnesses)
        avg_fit = sum(fitnesses) / len(fitnesses)

        best_fitness_per_gen.append(best_fit)
        avg_fitness_per_gen.append(avg_fit)

        if best_fit > global_best_fit:
            global_best_fit = best_fit
            global_best_chrom = population[fitnesses.index(best_fit)][:]

        # формируем новое поколение
        new_population: List[List[int]] = []

        # элитизм: переносим лучшего без изменений
        new_population.append(global_best_chrom[:])

        while len(new_population) < config.population_size:
            # селекция родителей
            parent1 = tournament_selection(population, fitnesses, config.tournament_size)
            parent2 = tournament_selection(population, fitnesses, config.tournament_size)

            # кроссовер
            child1, child2 = one_point_crossover(
                parent1, parent2, config.crossover_prob
            )

            # мутация
            mutate(child1, config.mutation_prob)
            if len(new_population) < config.population_size:
                mutate(child2, config.mutation_prob)
                new_population.append(child1)
                if len(new_population) < config.population_size:
                    new_population.append(child2)
            else:
                new_population.append(child1)

        population = new_population[:config.population_size]

    return GAHistory(
        best_fitness_per_gen=best_fitness_per_gen,
        avg_fitness_per_gen=avg_fitness_per_gen,
        best_chromosome=global_best_chrom,
        best_fitness=global_best_fit,
    )


# Форматирование расписания для вывода

def format_schedule(
    chromosome: List[int],
    lessons: List[Lesson]
) -> Dict[str, Dict[str, List[str]]]:
    """
    Удобное представление расписания:
    result[group][day] = список строк по парам.
    """
    result: Dict[str, Dict[str, List[str]]] = {}
    groups = sorted({lesson.group for lesson in lessons})
    for g in groups:
        result[g] = {}
        for d in DAYS:
            result[g][d] = ["-" for _ in range(SLOTS_PER_DAY)]

    for gene, slot_id in enumerate(chromosome):
        lesson = lessons[gene]
        day_idx = slot_id // SLOTS_PER_DAY
        pair_idx = slot_id % SLOTS_PER_DAY
        day_name = DAYS[day_idx]
        cell = f"{lesson.name} ({lesson.teacher})"
        if result[lesson.group][day_name][pair_idx] == "-":
            result[lesson.group][day_name][pair_idx] = cell
        else:
            # если конфликт, пометим явно
            result[lesson.group][day_name][pair_idx] += " | CONFLICT"

    return result
