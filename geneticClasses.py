# coding=utf-8
import numpy as np
import bge
import random
scene = bge.logic.getCurrentScene()

# variable global para setear la cantidad de sensores
sensors = 10
# variable global para setear la cantidad de salidas
outputs = 2
# variable global que setea la cantidad de neuronas de la capa oculta
first_hidden_layer_size = 15
# variable global que setea la cantidad de checkpoints que hay en total en la pista
total_checkpoints = 131

def sigmoid(x):
    '''

    :param x:
    :return:
    '''
    return (1/(1+np.exp(-x)))

def relu(x):
    '''

    :param x:
    :return:
    '''
    return x * (x > 0)


def crossParams(fatherParams,motherParams):
    newparams = []
    for i,obj in enumerate(fatherParams):
        if np.random.uniform(0., 1.) > .5:
            newparams.append(fatherParams[i])
           # print(fatherParams[i].__name__,'----------------------------')
           # print(fatherParams[i])
        else:
            newparams.append(motherParams[i])
    return newparams


def crossMutationParams(fatherParams,motherParams):
    newparams = []
    for i,obj in enumerate(fatherParams):
        random = np.random.uniform(0., 1.)
        if random > .66:
            newparams.append(fatherParams[i])
        elif random < .66 and random >.33:
            newparams.append(motherParams[i])
        else:
            newparams.append(np.random.uniform(-1.0, 1.0))
    return newparams


class Population():
    def __init__(self,max_iter,max_generations,initial_generation_size):
        '''

        :param max_iter:
        :param max_generations:
        :param initial_generation_size:
        '''
        self.max_iter = max_iter
        self.max_generations = max_generations
        self.generation_number = 0
        self.generation = [Individual(self.generation_number) for i in range(initial_generation_size)]
        for i in self.generation:
            i.physical = scene.addObject("car", "car")


    def evaluateGeneration(self):
        '''
        Funcion que evalua cada individuo de una generación y le asigna una determinada probabilidad de selección a cada
        uno, dependiendo de cuandos checkpoints haya alcanzado ese individuo.
        :return: None
        '''
        for i in range(len(self.generation)):
            test_individual = self.generation[i]
            '''
            Agregar acá logica para probar cada individuo con el modelo de blender!!
            '''
            
            #self.generation[i].selection_probability = checkpoints / total_checkpoints
            self.generation[i].selection_probability = self.generation[i].physical['waypoints'] / total_checkpoints

    def removePhysicalsGeneration(self):
        '''

        quisto90: First: delete all physical objects from previous generation
        '''
        for i in range(len(self.generation)):
            self.generation[i].removeIndividual()
            #print('removed: ',self.generation[i].name)

    def generateNewGeneration(self,mode):

        self.generation_number += 1
        new_generation = []
        for i in range(len(self.generation)):
            # Seleccion de los padres para cada individuo de la nueva generacion
            if mode == 'original':
                posible_fathers = []
                adaptative_threshold = 1.0
                while len(posible_fathers) < 2:
                    threshold = np.random.uniform(0, 1.0 * adaptative_threshold)
                    posible_fathers = [ind for ind in self.generation if (ind.selection_probability > threshold)]
                    adaptative_threshold -= 0.1
                father_index = int(np.random.randint(0, len(posible_fathers)))
                mother_index = int(np.random.randint(0, len(posible_fathers)))
                while mother_index == father_index:
                    mother_index = int(np.random.randint(0, len(posible_fathers)))
            if mode == 'new':
                # genero vector con acumulados de probabilidad
                a = []
                for i, obj in enumerate(self.generation):
                    if i - 1 < 0:
                        a.append(self.generation[i].selection_probability)
                    elif i + 1 > len(self.generation):
                        break
                    else:
                        a.append(self.generation[i].selection_probability + a[i - 1])
                # print('vector acumulado: ', a)

                # selecciono los padres:
                parents_indexes = []
                for i in range(2):
                    r = np.random.uniform(0, a[-1])
                    for i, obj in enumerate(a):
                        if r < a[0]:
                            parents_indexes.append(0)
                        elif r > a[i] and r < a[i + 1]:
                            parents_indexes.append(i + 1)
                            break
                if parents_indexes[0] == parents_indexes[1]:
                    parents_indexes[1] = parents_indexes[0]-1
                father_index = parents_indexes[0]
                mother_index = parents_indexes[1]

            # print('parents_index: ', father_index, mother_index)
            father = self.generation[father_index]
            mother = self.generation[mother_index]

            new_generation.append(self.createSon(self.generation_number,father,mother,mode="cross"))

        for j in new_generation:
            if j.generation_number == self.generation_number:
                # print('se crea fisico para ', j.name)
                j.addPhysical()

        self.generation = new_generation

        return

    def createSon(self,generation_number,father,mother,mode):
        '''
        Funcion que crea un hijo dependiendo de los atributos de los padres.
        :param generation_number:
        :param father: individuo padre
        :param mother: individuo madre
        :param mode: variable para controlar la forma en la cual se conbinan los padres
        :return: un individuo que contiene caracteristicas de ambos padres
        '''
        if mode == "mean":
            weights_1 = (np.array(father.weights_1) + np.array(mother.weights_1)) / 2
            bias_1 = (np.array(father.bias_1) + np.array(mother.bias_1)) / 2
            weights_2 = (np.array(father.weights_2) + np.array(mother.weights_2)) / 2
            bias_2 = (np.array(father.bias_2) + np.array(mother.bias_2)) / 2
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
            diff = np.array(father.weights_1)-np.array(mother.weights_1)
            weights_1 += np.random.uniform(0.,1.)*diff
            diff = np.array(father.bias_1) - np.array(mother.bias_1)
            bias_1 += np.random.uniform(0., 1.) * diff
            diff = np.array(father.weights_2) - np.array(mother.weights_2)
            weights_2 += np.random.uniform(0., 1.) * diff
            diff = np.array(father.bias_2) - np.array(mother.bias_2)
            bias_2 += np.random.uniform(0., 1.) * diff
        elif mode == "cross":
            weights_1 = crossParams(father.weights_1,mother.weights_1)
            bias_1 = crossParams(father.bias_1,mother.bias_1)
            weights_2 = crossParams(father.weights_2,mother.weights_2)
            bias_2 = crossParams(father.bias_2, mother.bias_2)
        elif mode == "crossMutation":
            weights_1 = crossMutationParams(father.weights_1,mother.weights_1)
            bias_1 = crossMutationParams(father.bias_1,mother.bias_1)
            weights_2 = crossMutationParams(father.weights_2,mother.weights_2)
            bias_2 = crossMutationParams(father.bias_2, mother.bias_2)
        else:
            weights_1 = None
            bias_1 = None
            weights_2 = None
            bias_2 = None

        return Individual(generation_number,weights_1,bias_1,weights_2,bias_2)

    def applyRandomMutation(self):
        '''

        :return:
        '''
        return

    def checkGeneration(self):
        '''
        Funcion que controla si ya se logro encontrar un hijo óptimo
        :return: None si no hay ningún hijo perfecto, el individuo si lo hay.
        '''
        for i in range(len(self.generation)):
            if self.generation[i].selection_probability == 1.:
                return self.generation[i]

        return None

    def getBestGeneration(self):
        '''

        :return:
        '''
        best_prob = self.generation[0].selection_probability
        best_index = 0
        for i in range(len(self.generation)):
            if self.generation[i].selection_probability >= best_prob:
                best_prob = self.generation[i].selection_probability
                best_index = i

        return self.generation[best_index]


class Individual():
    def __init__(self,generation_number,weights_1 = None,bias_1 = None,weights_2 = None,bias_2 = None):
        '''

        :param weights_1: pesos de la primera capa para inicializacion
        :param bias_1: bias de la primera capa para inicializacion
        :param weights_2: pesos de la segunda capa para inicializacion
        :param bias_2: bias de la segunda capa para inicializacion

        '''

        if not np.all(weights_1):
            self.weights_1 = np.random.uniform(-1,1,[sensors,first_hidden_layer_size])
            self.bias_1 = np.random.uniform(-1,1,[1,first_hidden_layer_size])
            self.weights_2 = np.random.uniform(-1, 1, [first_hidden_layer_size,outputs])
            self.bias_2 = np.random.uniform(-1, 1, [1,outputs])
        else:
            self.weights_1 = weights_1
            self.bias_1 = bias_1
            self.weights_2 = weights_2
            self.bias_2 = bias_2

        self.layer1_activation_function = "relu"
        self.selection_probability = 0.0
        self.generation_number = generation_number
        self.name = ''.join(random.choice('0123456789ABCDEF') for i in range(16))
        # print('individuo creado: ',self.name)

    def calculateOutputs(self,sensor_inputs):
        '''

        :param sensor_inputs: entradas de los sensores (vector de unos y ceros)
        :return: steer en un rango de (:) y accelerator en un rango de (:)
        '''
        first_layer_activation = np.dot(sensor_inputs,self.weights_1)+self.bias_1
        if self.layer1_activation_function == "relu":
            first_layer_activation = relu(first_layer_activation)
        elif self.layer1_activation_function == "sigmoid":
            first_layer_activation = sigmoid(first_layer_activation)

        output_layer_activation = np.dot(first_layer_activation ,self.weights_2) + self.bias_2
        steer = output_layer_activation[0,0]
        accelerator = sigmoid(output_layer_activation[0,1])

        return [steer,accelerator]

    def getHistograms(self):
        weights1_dist, bin_edges = np.histogram(self.weights_1,bins=100,range=(-1,1))
        weights2_dist, bin_edges = np.histogram(self.weights_2, bins=100, range=(-1, 1))
        bias1_dist, bin_edges = np.histogram(self.bias_1, bins=100, range=(-1, 1))
        bias2_dist, bin_edges = np.histogram(self.bias_2, bins=100, range=(-1, 1))
        return [weights1_dist,bias1_dist,weights2_dist,bias2_dist]

    def removeIndividual(self):
        self.physical.endObject()

    def addPhysical(self):
        # quisto90: First: create physical
        self.physical = scene.addObject("car", "car")
        #self.physical['name']= 'car' + str(index)
        #self.name = 'car'+str(index)
