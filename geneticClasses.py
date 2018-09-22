import numpy as np
import copy


class Genetic():
    def __init__(self,generationSize, sensorSize, firstHiddenLayerSize, outputSize, proportion_a, proportion_b):
        self.generationSize = generationSize
        self.generationNumber = 0
        self.generations = [Generation(self.generationNumber)]
        self.sensorSize = sensorSize
        self.outputSize = outputSize
        self.firstHiddenLayerSize = firstHiddenLayerSize
        self.proportion_a = proportion_a
        self.proportion_b = proportion_b
        self.proportion_c = 1 - (proportion_a + proportion_b)
        self.create_first_geneation()

    def create_first_geneation(self):
        for i in range(self.generationSize):
            weights_1 = np.random.uniform(-1, 1, [self.sensorSize, self.firstHiddenLayerSize])
            bias_1 = np.random.uniform(-1, 1, [1, self.firstHiddenLayerSize])
            weights_2 = np.random.uniform(-1, 1, [self.firstHiddenLayerSize, self.outputSize])
            bias_2 = np.random.uniform(-1, 1, [1, self.outputSize])
            self.generations[0].individuals.append(Individual(self.generations[0], weights_1, bias_1, weights_2, bias_2))

    def create_new_generation(self, mode):
        self.generationNumber += 1
        self.generations[-1].set_max_strength()
        self.generations[-1].set_all_selection_probability()
        self.set_individual_types()
        self.generations.append(Generation(self.generationNumber))
        if len(self.generations) >= 3:
            del self.generations[0]
        print('hay '+ str(len(self.generations))+'anteriores')
        print('---------- se crea la generación: ' + str(self.generationNumber) + '----------------------------')
        for individual in sorted(self.generations[-2].individuals, key=lambda ind: ind.strength, reverse=True):
            print(id(individual), individual.strength, individual.type, individual.selection_probability)


        for individual in self.generations[-2].individuals:
            if individual.type == 'a':
                self.generations[-1].individuals.append(copy.copy(individual))
            elif individual.type == 'b':
                sorted_parents = sorted(self.generations[-2].individuals, key=lambda ind: ind.strength, reverse=True)
                parents_indexes = self.get_probable_fittest_parents(sorted_parents)
                father_index = parents_indexes[0]
                mother_index = parents_indexes[1]
                father = sorted_parents[father_index]
                mother = sorted_parents[mother_index]
                self.generations[-1].individuals.append(self.create_son(father, mother))
            else:
                sorted_parents = sorted(self.generations[-2].individuals, key=lambda ind: ind.strength, reverse=True)
                parents_indexes = self.get_probable_fittest_parents(sorted_parents)
                mother_index = parents_indexes[1]
                mother = sorted_parents[mother_index]
                weights_1 = np.random.uniform(-1, 1, [self.sensorSize, self.firstHiddenLayerSize])
                bias_1 = np.random.uniform(-1, 1, [1, self.firstHiddenLayerSize])
                weights_2 = np.random.uniform(-1, 1, [self.firstHiddenLayerSize, self.outputSize])
                bias_2 = np.random.uniform(-1, 1, [1, self.outputSize])
                father = Individual(self.generations[0], weights_1, bias_1, weights_2, bias_2)
                self.generations[-1].individuals.append(self.create_son(father, mother))

    def get_probable_fittest_parents(self, individuals):
        accumulated_prob = []
        for i, obj in enumerate(individuals):
            if i - 1 < 0:
                accumulated_prob.append(individuals[i].selection_probability)
            elif i + 1 > len(individuals):
                break
            else:
                accumulated_prob.append(individuals[i].selection_probability + accumulated_prob[i - 1])


        parents_indexes = []
        for i in range(2):
            r = np.random.uniform(0, accumulated_prob[-1])
            for j, obj in enumerate(accumulated_prob):
                if r <= accumulated_prob[0]:
                    parents_indexes.append(0)
                elif accumulated_prob[j] < r < accumulated_prob[j + 1]:
                    parents_indexes.append(j + 1)
                    break
        if parents_indexes[0] == parents_indexes[1]:
            parents_indexes[1] = parents_indexes[0] - 1

        return parents_indexes

    def set_individual_types(self):
        sorted_best_individuals = sorted(self.generations[-1].individuals, key=lambda ind: ind.strength, reverse=True)
        for i, individual in enumerate(sorted_best_individuals):
            if i/self.generationSize <= self.proportion_a:
                individual.type = 'a'
            if self.proportion_a < i / self.generationSize <= (self.proportion_a + self.proportion_b):
                individual.type = 'b'
            if (self.proportion_a + self.proportion_b) < i / self.generationSize < (self.proportion_a + self.proportion_b + self.proportion_c):
                individual.type = 'c'

    def create_son(self, father, mother):
        mode = "cross_params"
        if mode == "cross_params":
            weights_1 = self.cross_params(father.weights_1, mother.weights_1)
            bias_1 = self.cross_params(father.bias_1, mother.bias_1)
            weights_2 = self.cross_params(father.weights_2, mother.weights_2)
            bias_2 = self.cross_params(father.bias_2, mother.bias_2)
        elif mode == "diff":
            if father.selection_probability > mother.selection_probability:
                weights_1 = father.weights_1
                weights_2 = father.weights_2
                bias_1 = father.bias_1
                bias_2 = father.bias_2
            else:
                weights_1 = mother.weights_1
                weights_2 = mother.weights_2
                bias_1 = mother.bias_1
                bias_2 = mother.bias_2
            diff = np.array(father.weights_1) - np.array(mother.weights_1)
            weights_1 += np.random.uniform(0., 1.) * diff
            diff = np.array(father.bias_1) - np.array(mother.bias_1)
            bias_1 += np.random.uniform(0., 1.) * diff
            diff = np.array(father.weights_2) - np.array(mother.weights_2)
            weights_2 += np.random.uniform(0., 1.) * diff
            diff = np.array(father.bias_2) - np.array(mother.bias_2)
            bias_2 += np.random.uniform(0., 1.) * diff
        elif mode == "mean":
            weights_1 = (np.array(father.weights_1) + np.array(mother.weights_1)) / 2
            bias_1 = (np.array(father.bias_1) + np.array(mother.bias_1)) / 2
            weights_2 = (np.array(father.weights_2) + np.array(mother.weights_2)) / 2
            bias_2 = (np.array(father.bias_2) + np.array(mother.bias_2)) / 2
        return Individual(self.generations[-1], weights_1, bias_1, weights_2, bias_2)

    def cross_params(self, father_params, mother_params):
        new_params = []
        for i, obj in enumerate(father_params):
            if np.random.uniform(0., 1.) > .5:
                new_params.append(father_params[i])
            else:
                new_params.append(mother_params[i])
        return new_params

    def sigmoid(x):
        return (1 / (1 + np.exp(-x)))

    def relu(x):
        return x * (x > 0)


class Generation:
    def __init__(self, generationNumber):
        self.generationNumber = generationNumber
        self.individuals = []
        self.max_strength = 0.0

    def are_all_dead(self):
        allDead = True
        for individual in self.individuals:
            # print(self.generationNumber, id(individual),individual.alive)
            if individual.alive:
                allDead = False
                break
        return allDead

    def set_max_strength(self):
        for individual in self.individuals:
            if individual.strength >= self.max_strength:
                self.max_strength = individual.strength

    def set_all_selection_probability(self):
        for individual in self.individuals:
            individual.set_selection_probability()



class Individual:
    def __init__(self, generation, weights_1 = None,bias_1 = None,weights_2 = None,bias_2 = None):
        self.weights_1 = weights_1
        self.bias_1 = bias_1
        self.weights_2 = weights_2
        self.bias_2 = bias_2
        self.generation = generation
        self.alive = True
        self.strength = 0.0
        self.selection_probability = 0.0
        self.layer1_activation_function = "relu"

    def calculateOutputs(self, sensor_inputs):

        first_layer_activation = np.dot(sensor_inputs, self.weights_1) + self.bias_1
        if self.layer1_activation_function == "relu":
            first_layer_activation = Genetic.relu(first_layer_activation)
        elif self.layer1_activation_function == "sigmoid":
            first_layer_activation = Genetic.sigmoid(first_layer_activation)

        output_layer_activation = np.dot(first_layer_activation, self.weights_2) + self.bias_2
        # cambiar por extraer columna

        steer = output_layer_activation[0, 0]
        accelerator = Genetic.sigmoid(output_layer_activation[0, 1])

        return [steer, accelerator]

    def set_selection_probability(self):
        if self.generation.max_strength != 0.0:
            self.selection_probability = self.strength / self.generation.max_strength

