# coding=utf-8
import numpy as np
import bge
scene = bge.logic.getCurrentScene()

# variable global para setear la cantidad de sensores
sensors = 10;
# variable global para setear la cantidad de salidas
outputs = 2;
# variable global que setea la cantidad de neuronas de la capa oculta
first_hidden_layer_size = 15;
# variable global que setea la cantidad de checkpoints que hay en total en la pista
total_checkpoints = 131;

def sigmoid(x):
    '''

    :param x:
    :return:
    '''
    return (1/(1+np.exp(-x)));

def relu(x):
    '''

    :param x:
    :return:
    '''
    return x * (x > 0);

class Population():
    def __init__(self,max_iter,max_generations,initial_generation_size):
        '''

        :param max_iter:
        :param max_generations:
        :param initial_generation_size:
        '''
        self.max_iter = max_iter
        self.max_generations = max_generations
        self.generation = [Individual() for i in range(initial_generation_size)]
        self.generation_number = 0

    def evaluateGeneration(self):
        '''
        Funcion que evalua cada individuo de una generación y le asigna una determinada probabilidad de selección a cada
        uno, dependiendo de cuandos checkpoints haya alcanzado ese individuo.
        :return: None
        '''
        for i in range(len(self.generation)):
            test_individual = self.generation[i];
            '''
            Agregar acá logica para probar cada individuo con el modelo de blender!!
            '''
            
            #self.generation[i].selection_probability = checkpoints / total_checkpoints;
            self.generation[i].selection_probability = self.generation[i].physical['waypoints'] / total_checkpoints

    def generateNewGeneration(self):
        '''

        quisto90: First: delete all physical objects from previous generation
        '''
        for i in range(len(self.generation)):
            self.generation[i].physical.endObject()

        new_generation = [Individual() for i in range(len(self.generation))];
        for i in range(len(self.generation)):
            # Seleccion de los padres para cada individuo de la nueva generacion
            posible_fathers = [];
            adaptative_threshold = 1.;
            while len(posible_fathers) < 2:
                threshold = np.random.uniform(0,1.*adaptative_threshold);
                posible_fathers = [ind for ind in self.generation if (ind.selection_probability > threshold)];
                adaptative_threshold -= 0.1;
            father_index = np.random.randint(0,len(posible_fathers));
            mother_index = np.random.randint(0, len(posible_fathers));
            while mother_index == father_index:
                mother_index = np.random.randint(0, len(posible_fathers));

            father = self.generation[father_index];
            mother = self.generation[mother_index];

            new_generation[i] = self.createSon(father,mother,mode="mean");

        self.generation = new_generation;
        self.generation_number += 1;
        return;

    def createSon(self,father,mother,mode="mean"):
        '''
        Funcion que crea un hijo dependiendo de los atributos de los padres.
        :param father: individuo padre
        :param mother: individuo madre
        :param mode: variable para controlar la forma en la cual se conbinan los padres
        :return: un individuo que contiene caracteristicas de ambos padres
        '''
        if mode == "mean":
            weights_1 = np.mean([father.weights_1, mother.weights_1]);
            bias_1 = np.mean([father.bias_1, mother.bias_1]);
            weights_2 = np.mean([father.weights_2, mother.weights_2]);
            bias_2 = np.mean([father.bias_2, mother.bias_2]);
        else:
            weights_1 = None;
            bias_1 = None;
            weights_2 = None;
            bias_2 = None;

        return Individual(weights_1,bias_1,weights_2,bias_2);

    def applyRandomMutation(self):
        '''

        :return:
        '''
        return;

    def checkGeneration(self):
        '''
        Funcion que controla si ya se logro encontrar un hijo óptimo
        :return: None si no hay ningún hijo perfecto, el individuo si lo hay.
        '''
        for i in range(len(self.generation)):
            if self.generation[i].selection_probability == 1.:
                return self.generation[i];

        return None;

    def getBestGeneration(self):
        '''

        :return:
        '''
        best_prob = self.generation[0].selection_probability;
        best_index = 0;
        for i in range(len(self.generation)):
            if self.generation[i].selection_probability >= best_prob:
                best_prob = self.generation[i].selection_probability;
                best_index = i;

        return self.generation[best_index];


class Individual():
    def __init__(self,weights_1 = None,bias_1 = None,weights_2 = None,bias_2 = None):
        '''

        :param weights_1: pesos de la primera capa para inicializacion
        :param bias_1: bias de la primera capa para inicializacion
        :param weights_2: pesos de la segunda capa para inicializacion
        :param bias_2: bias de la segunda capa para inicializacion

        '''

        if not weights_1:
            self.weights_1 = np.random.uniform(-1,1,[sensors,first_hidden_layer_size]);
            self.bias_1 = np.random.uniform(-1,1,[1,first_hidden_layer_size]);
            self.weights_2 = np.random.uniform(-1, 1, [first_hidden_layer_size,outputs]);
            self.bias_2 = np.random.uniform(-1, 1, [1,outputs]);
        else:
            self.weights_1 = weights_1;
            self.bias_1 = bias_1;
            self.weights_2 = weights_2;
            self.bias_2 = bias_2;

        self.layer1_activation_function = "relu";
        self.selection_probability = 0.0;

	#quisto90: First: create physical
        self.physical = scene.addObject("car","car")

    def calculateOutputs(self,sensor_inputs):
        '''

        :param sensor_inputs: entradas de los sensores (vector de unos y ceros)
        :return: steer en un rango de (:) y accelerator en un rango de (:)
        '''
        first_layer_activation = np.dot(sensor_inputs,self.weights_1)+self.bias_1;
        if self.layer1_activation_function == "relu":
            first_layer_activation = relu(first_layer_activation);
        elif self.layer1_activation_function == "sigmoid":
            first_layer_activation = sigmoid(first_layer_activation);

        output_layer_activation = np.dot(first_layer_activation ,self.weights_2) + self.bias_2;
        steer = output_layer_activation[0,0];
        accelerator = sigmoid(output_layer_activation[0,1]);

        return [steer,accelerator];


