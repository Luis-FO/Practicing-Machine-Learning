"""
This code was the reproduction of this other one: https://www.geeksforgeeks.org/genetic-algorithms/
It was rewritten by me just for personal learning about the concept of GA's, but it's 
the same code.
"""

import random

POPULATION_SIZE = 100

GENES = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP 
QRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}'''

TARGET = "I AM LEARNING GA" 

class Individual():

    def __init__(self, chromosome) -> None:
        self.chromosome = chromosome
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        fit = 0
        return fit
    
    @classmethod
    def mutated_genes(self):
        global GENES
        return random.choice(GENES)
    
    @classmethod
    def create_gnome(self):
        global TARGET
        target_gnome_size = len(TARGET)
        return [self.mutated_genes() for _ in range(target_gnome_size)]
    
    def calculate_fitness(self):
        global TARGET
        fitness = 0
        for gs, gt in zip(self.chromosome, TARGET):
            if gs != gt:
                fitness+=1

        return fitness
    def mate(self, parent2):

        child_chromosome = []

        for gparent1, gparent2 in zip(self.chromosome, parent2.chromosome):

            #[0, 1)
            probability = random.random()

            if probability < 0.45:
                child_chromosome.append(gparent1)
            elif probability < 0.90:
                child_chromosome.append(gparent2)
            else:
                child_chromosome.append(self.mutated_genes())

        return Individual(child_chromosome)
    
    def __str__(self) -> str:
        return "".join(self.chromosome)


def main():
    
    global POPULATION_SIZE

    generation_num = 1

    found = False

    population = []

    for _ in range(POPULATION_SIZE):
        gnome = Individual.create_gnome()
        population.append(Individual(gnome))

    while not found:

        # sort population
        # Key is the atribute fitness of Individual
        population = sorted(population, key = lambda x:x.fitness)

        if population[0].fitness == 0:
            found = True
            break

        new_generation = [] 
        
        # Elitism 10%
        s = int((10*POPULATION_SIZE)/100)
        new_generation.extend(population[:s])

        s = int((90*POPULATION_SIZE)/100)

        for _ in range(s):
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            child = parent1.mate(parent2)
            new_generation.append(child)
        
        population = new_generation
        
        print(f"Generation: {generation_num}\tString: {population[0].chromosome}\tFitness: {population[0].fitness}") 

        generation_num+=1

    print(f"Generation: {generation_num}\tString: {population[0].chromosome}\tFitness: {population[0].fitness}") 

if __name__ == "__main__":
    main()