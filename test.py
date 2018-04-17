import geneticClasses as myClasses
import numpy as np

ind = myClasses.Individual();
inputs = np.asarray([[0,1,1,1,0,0,0,0,0,0]]);
print("Output: "+ str(ind.calculateOutputs(inputs)));