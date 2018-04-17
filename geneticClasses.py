import numpy as np

sensors = 10;
outputs = 2;
first_hidden_layer_size = 15;
total_checkpoints = 100;

def sigmoid(x):
    return (1/(1+np.exp(-x)));

def relu(x):
    return x * (x > 0);

class Population():
    def __init__(self,max_iter,max_generations,initial_generation_size):
        self.max_iter = max_iter;
        self.max_generations = max_generations;
        self.generation = [Individual() for i in range(initial_generation_size)];

    def evaluateGeneration(self):
        for i in range(len(self.generation)):
            test_individual = self.generation[i];
            '''
            Agregar logica para probar cada red con el modelo!!
            Meterle a la funcion calculateOutput de test_individual los sensores (en formato unos y ceros). Devuelve un
            tuple donde el primer elemento es steer [] y el segundo el acelerador ()
            '''
            self.generation[i].selection_probability = checkpoints / total_checkpoints;

    def generateNewGeneration(self):
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

            weights_1 = np.mean([father.weights_1,mother.weights_1]);
            bias_1 = np.mean([father.bias_1, mother.bias_1]);
            weights_2 = np.mean([father.weights_2, mother.weights_2]);
            bias_2 = np.mean([father.bias_2, mother.bias_2]);

            new_generation[i].weights_1 = np.copy(weights_1);
            new_generation[i].bias_1 = np.copy(bias_1);
            new_generation[i].weights_2 = np.copy(weights_2);
            new_generation[i].bias_2 = np.copy(bias_2);

        self.generation = new_generation;
        return;

    def applyRandomMutation(self):
        return;

    def checkGeneration(self):
        for i in range(len(self.generation)):
            if self.generation[i].selection_probability == 1.:
                return self.generation[i];

        return None;

    def getBestGeneration(self):
        best_prob = self.generation[0].selection_probability;
        best_index = 0;
        for i in range(len(self.generation)):
            if self.generation[i].selection_probability >= best_prob:
                best_prob = self.generation[i].selection_probability;
                best_index = i;

        return self.generation[best_index];


class Individual():
    def __init__(self,weights_1 = None,bias_1 = None,weights_2 = None,bias_2 = None):
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

    def calculateOutputs(self,sensor_inputs):
        first_layer_activation = sensor_inputs*self.weights_1+self.bias_1;
        if self.layer1_activation_function == "relu":
            first_layer_activation = relu(first_layer_activation);
        elif self.layer1_activation_function == "sigmoid":
            first_layer_activation = sigmoid(first_layer_activation);

        output_layer_activation = first_layer_activation * self.weights_2 + self.bias_2;
        steer = output_layer_activation[0];
        accelerator = sigmoid(output_layer_activation[1]);

        return [steer,accelerator];


