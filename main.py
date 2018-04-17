import numpy as np
import geneticClasses as myClasses
import sys
import cPickle as pickle


def main():
    # Read parameters
    max_iter = sys.argv[1];
    max_generations = sys.argv[2];
    initial_generation_size = sys.argv[3];
    mutation_probability = sys.argv[4];

    # Generate population
    population = myClasses.Population(max_iter,max_generations,initial_generation_size);

    # Search for optimal solution
    for i in range(max_iter):
        population.evaluateGeneration();
        solution = population.checkGeneration();
        if not solution:
            population.generateNewGeneration();
        else:
            break;

        population.applyRandomMutation(mutation_probability);

    # If there is no perfect solution, look for the one with the highest selection probability
    if not solution:
        solution = population.getBestGeneration();

    print("Highest selection probability: " + str(solution.selection_probability));
    print("Saving model...");

    with open("model.pkl","wb") as f:
        pickle.dump(solution,f);


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Sintax is:");
        print("python main.py <max_iter> <max_generations> <initial_generation_size>");
    else:
        main();
        print("End.");