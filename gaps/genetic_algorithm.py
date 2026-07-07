from __future__ import print_function

import random

from operator import attrgetter

from gaps import utils
from gaps.crossover import Crossover
from gaps.image_analysis import ImageAnalysis
from gaps.individual import Individual
from gaps.plot import Plot
from gaps.progress_bar import print_progress
from gaps.selection import roulette_selection


class GeneticAlgorithm(object):
    TERMINATION_THRESHOLD = 10

    def __init__(
        self,
        image,
        piece_size,
        population_size,
        generations,
        elite_size=2,
        mutation_rate=0.0,
    ):
        self._image = image
        self._piece_size = piece_size
        self._generations = generations
        self._elite_size = elite_size
        self._mutation_rate = mutation_rate
        pieces, rows, columns = utils.flatten_image(image, piece_size, indexed=True)
        self._population = [
            Individual(pieces, rows, columns) for _ in range(population_size)
        ]
        self._pieces = pieces
        self.generations_completed = 0
        self.termination_reason = None
        self.best_fitness = None
        self.fitness_history = []

    def start_evolution(self, verbose, generation_callback=None):
        print("=== Pieces:      {}\n".format(len(self._pieces)))
        self.generations_completed = 0
        self.termination_reason = None
        self.best_fitness = None
        self.fitness_history = []

        if verbose:
            plot = Plot(self._image)

        ImageAnalysis.analyze_image(self._pieces)

        fittest = None
        best_fitness_score = float("-inf")
        termination_counter = 0

        for generation in range(self._generations):
            print_progress(
                generation, self._generations - 1, prefix="=== Solving puzzle: "
            )

            new_population = []

            # Elitism
            elite = self._get_elite_individuals(elites=self._elite_size)
            new_population.extend(elite)

            selected_parents = roulette_selection(
                self._population, elites=self._elite_size
            )

            for first_parent, second_parent in selected_parents:
                crossover = Crossover(first_parent, second_parent)
                crossover.run()
                child = crossover.child()
                if random.random() < self._mutation_rate:
                    self._mutate(child)
                new_population.append(child)

            self._population = new_population
            fittest = self._best_individual()
            self.generations_completed = generation + 1
            self._record_fitness_history(self.generations_completed)
            if generation_callback is not None:
                generation_callback(self.generations_completed, fittest)

            if fittest.fitness <= best_fitness_score:
                termination_counter += 1
            else:
                best_fitness_score = fittest.fitness
                self.best_fitness = best_fitness_score
                termination_counter = 0

            if termination_counter == self.TERMINATION_THRESHOLD:
                print("\n\n=== GA terminated")
                print(
                    "=== There was no improvement for {} generations".format(
                        self.TERMINATION_THRESHOLD
                    )
                )
                self.termination_reason = "stagnation"
                return fittest

            if verbose:
                plot.show_fittest(
                    fittest.to_image(),
                    "Generation: {} / {}".format(generation + 1, self._generations),
                )

        self.termination_reason = "max_generations"
        return fittest

    def _get_elite_individuals(self, elites):
        """Returns first 'elite_count' fittest individuals from population"""
        return sorted(self._population, key=attrgetter("fitness"))[-elites:]

    def _best_individual(self):
        """Returns the fittest individual from population"""
        return max(self._population, key=attrgetter("fitness"))

    def _record_fitness_history(self, generation):
        fitness_values = [individual.fitness for individual in self._population]
        self.fitness_history.append(
            {
                "generation": generation,
                "best_fitness": max(fitness_values),
                "average_fitness": sum(fitness_values) / len(fitness_values),
                "worst_fitness": min(fitness_values),
            }
        )

    def _mutate(self, individual):
        """Swaps two random pieces in an individual."""
        if len(individual.pieces) < 2:
            return

        first_index, second_index = random.sample(range(len(individual.pieces)), 2)
        individual.pieces[first_index], individual.pieces[second_index] = (
            individual.pieces[second_index],
            individual.pieces[first_index],
        )
        individual._piece_mapping = {
            piece.id: index for index, piece in enumerate(individual.pieces)
        }
        individual._fitness = None
