import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import ga


#Sphere Test Function
def sphere(x):
    return sum(x**2)

#Problem Definition
problem = structure()
problem.costfunc = sphere
problem.nvar = 5 #number of variable
problem.varmin = -10 #lower bound of variable
problem.varmax = 10 #highest bound of variable

#GA Parameter
params = structure()
params.max_iteration = 100
params.n_pop = 20
params.proportion_children = 1
params.gamma = 0.1
params.mu = 0.1
params.sigma = 0.1
params.beta = 1

#Run GA
out = ga.run(problem, params)

#Result
# plt.plot(out.best_cost)
plt.semilogy(out.best_cost)
plt.xlim(0, params.max_iteration)
plt.xlabel('Iterations')
plt.ylabel('Best Cost')
plt.title('Genetic Algorithm (GA)')
plt.grid(True)
plt.show()