import geneticClasses
import bge
scene = bge.logic.getCurrentScene()


class BlenderGenetic(geneticClasses.Genetic):
    def __init__(self, generation_size, sensor_size, first_hidden_layer_size, proportion_a, proportion_b):
        self.outputSize = 2
        geneticClasses.Genetic.__init__(self, generation_size, sensor_size, first_hidden_layer_size, self.outputSize, proportion_a, proportion_b)
        self.create_first_generation_physicals()

    def create_first_generation_physicals(self):
        for individual in self.generations[0].individuals:
            individual.physical = BlenderPhysical(individual)

    def calculate_all_movement(self):
        for individual in self.generations[-1].individuals:
            individual.physical.calculate_movement()

    def read_individuals(self):
        for individual in self.generations[-1].individuals:
            individual.alive = individual.physical.get_alive()
            individual.strength = individual.physical.get_strength()
            # print(individual.alive, individual.strength)

    def create_new_generation_physicals(self):
        geneticClasses.Genetic.create_new_generation(self, mode='')
        for individual in self.generations[-1].individuals:
            individual.physical = BlenderPhysical(individual)

    def remove_all_generation_physicals(self, generation):
        for individual in generation.individuals:
            individual.physical.remove_individual()


class BlenderPhysical:
    # def __init__(self, *args, **kwargs):
    #     super(BlenderIndividual, self).__init__(*args, **kwargs)

    def __init__(self, individual):
        self.physical = scene.addObject("bug", "bug")
        self.individual = individual

    def calculate_movement(self):
        sensors = []
        for sensor in self.physical.children:
            if 'sensor' in sensor:
                sensors.append(self.physical[sensor.name])
        # print(sensors)
        self.physical['velocity'] = self.individual.calculateOutputs(sensors)[0]/10
        self.physical['turn'] = self.individual.calculateOutputs(sensors)[1]/100
        # print(self.physical['velocity'],self.physical['turn'] )

    def get_alive(self):
        return self.physical['alive']

    def get_strength(self):
        return self.physical['strength']

    def remove_individual(self):
        self.physical.endObject()
