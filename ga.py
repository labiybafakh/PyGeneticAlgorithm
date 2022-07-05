from math import gamma
from matplotlib.pyplot import flag
import numpy as np
from ypstruct import structure

def run(problem, params):

    #Problem Information
    costfunc = problem.costfunc
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax

    #Parameters
    max_iteration = params.max_iteration
    n_pop = params.n_pop
    proportion_children = params.proportion_children
    number_children = int(np.round(proportion_children * n_pop/2)*2)
    gamma = params.gamma
    mu = params.mu
    sigma = params.sigma
    beta = params.beta

    #Empty Individual Template
    empty_individual = structure()
    empty_individual.position = None
    empty_individual.cost = None

    #Best Solution Ever Found
    best_solution = empty_individual.deepcopy()
    best_solution.cost = np.inf

    #Initialize Population
    pop = empty_individual.repeat(n_pop)
    for i in range(0, n_pop):
        pop[i].position = np.random.uniform(varmin, varmax, nvar)
        pop[i].cost = costfunc(pop[i].position)

        if pop[i].cost < best_solution.cost:
            best_solution = pop[i].deepcopy()
    
    
    #Best Cost of Iterations
    best_cost = np.empty(max_iteration)

    #Main Loop
    for it in range(max_iteration):
        costs = np.array([x.cost for x in pop])
        average_cost = np.mean(costs)
        if average_cost != 0:
            costs = costs/average_cost
        probs = np.exp(-beta*costs)

        popc = []
        for _ in range(number_children//2):

            #Select Parents
            # q = np.random.permutation(n_pop)
            # p1 = pop[q[0]]
            # p2 = pop[q[1]]

            #Perform Roulette Wheel Selection
            p1 = pop[roulette_wheel_selection(probs)]
            p2 = pop[roulette_wheel_selection(probs)]

            #Performing Crossover
            c1, c2 = crossover(p1, p2)

            #Performing Mutation
            c1 = mutate(c1, mu, sigma)
            c2 = mutate(c2, mu, sigma)
            
            #Apply Bounds
            apply_bound(c1, varmin, varmax)
            apply_bound(c2, varmin, varmax)

            #Evaluate First offspring
            c1.cost = costfunc(c1.position)
            if c1.cost < best_solution.cost:
                best_solution = c1.deepcopy()
            
            #Evaluate Second offspring
            c2.cost = costfunc(c1.position)
            if c2.cost < best_solution.cost:
                best_solution = c2.deepcopy()

            #Add Offspring to popc
            popc.append(c1)
            popc.append(c2)

            #Merge, Sort, and Select
            pop += popc
            pop = sorted(pop, key=lambda x: x.cost)
            pop = pop[0:n_pop]

            #Store Best Cost
            best_cost[it] = best_solution.cost

            #Show Iteration Information
            print("Iteration {}: Best Cost = {}".format(it, best_cost[it]))


    #Output
    out = structure()
    out.pop = pop
    out.best_solution = best_solution
    out.best_cost = best_cost
    return out


def crossover(p1, p2, gamma=0.1):
    c1 = p1.deepcopy()
    c2 = p1.deepcopy()
    alpha = np.random.uniform(-gamma, 1+gamma, *c1.position.shape)
    c1.position = alpha*p1.position + (1-alpha)*p2.position
    c2.position = alpha*p2.position + (1-alpha)*p1.position
    return c1, c2

def mutate(x, mu, sigma):
    y = x.deepcopy()
    flag = np.random.rand(*x.position.shape) <= mu
    ind = np.argwhere(flag)
    y.position[ind] += sigma*np.random.rand(*ind.shape)
    return y

def apply_bound(x, varmin, varmax):
    x.position = np.maximum(x.position, varmin)
    x.position = np.minimum(x.position, varmax)

def  roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere(r<=c)
    return ind[0][0]