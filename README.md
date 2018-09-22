# AI Bugs

Open .blend file, modify  ai.py script line: own['bugs'] = myClasses.BlenderGenetic(50,10,15,0.1,0.3) as (generationSize, sensorSize, firstHiddenLayerSize, outputSize, proportion_a, proportion_b)
sensorSize must be 10
proportion_a is the proportion of individuals that won't change
proportion_b is the proportion of individuals that will cross each others
proportion_c = 1-(proportion_a+proportion_b) is the proportion of individuals that will cross with a randon one to mutate

create_son(self, father, mother) in geneticClasses.py is the function that cross parents.
