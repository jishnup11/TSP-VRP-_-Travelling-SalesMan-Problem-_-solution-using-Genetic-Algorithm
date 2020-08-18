#!/usr/bin/env python
# coding: utf-8

#  

# # Vehicle Routing Problem
# The vehicle routing problem (VRP) is a combinatorial optimization and integer 
# programming problem seeking to service a number of customers with a fleet of vehicles. 
# VRP is an important problem in the fields of transportation, distribution and logistics. Often 
# the context is that of delivering goods located at a central depot to customers who have 
# placed orders for such goods. Implicit is the goal of minimizing the cost of distributing the 
# goods. Many methods have been developed for searching for good solutions to the 
# problem, but finding global minimum for the cost function is computationally complex.
# 
# Design and develop a genotype model for representing VRP, and also develop an 
# Evolutionary Algorithm that can evolve candidate solutions and find the solution.

# We are considering the cost function as the distance covered to travel through all the cities exactly once.
# We are reducing the the VRP to travelling salesman problem by the assumption that
# * If we can reduce the distance travelled to cover the longest route, then we can then we can complete the each route with mininal time and there by accomplish delivery of larger number of orders by allocating multiple routes to vehicle which completes the delivery fast and there by we can reduce the cost and maximise the profit.
# ## Travelling salesman Problem
# 
# Constraints
# * Each city needs to be visited exactly one time
# * We must return to the starting city, so our total distance needs to be calculated accordingly
# 
# opimisation problem
# * Our objective is to minimize the distance travelled by saleseman(here vehicle) to travell through all the cities.

# In[ ]:





# # Design of Experiments
# GA

# * Representation	--------               Permutation of Integer tuples which represents the city
# 
# 
# * Fitness          --------   	           Inverse of the route distance
# 
# 
# * Recombination	     --------           ordered crossover, elitism size 20	
# 
# 
# * Mutation	           --------          Swap
# 
# 
# * Mutation Probability	 --------        0.01
# 
# 
# * Parent Selection	      --------       Best 2 
# 
# 
# * Survival Selection	      --------       rank, roulette wheel, elitislm 20
# 
# 
# * Population size	            --------     100
# 
# 
# * Number of Offspring	        --------     2
# 
# 
# * Initialization 	          --------       Random
# 
# 
# * Termination condition	     --------    500 generations

# In[ ]:





# In[ ]:





# # Program Development

# In[18]:


import numpy as np
import  random
import operator
import  pandas as pd
import matplotlib.pyplot as plt


# In[ ]:





# In[19]:



class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2)) #Euclidian distance
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


# In[20]:


# Fitness function

class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance()) # we are taking the inverse so that to perform minimisation of distance. Larger fitness permitted
        return self.fitness


# In[21]:


# Initial population
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route
def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population


# In[22]:


#rank base selection

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum() # Roulette wheel selection
    #print(df)
    for i in range(0, eliteSize):   #Elitism to make sure that the fittest individuals are permitted to next gen properly
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        #pick
        #print(df.iat[i,3])
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


# In[23]:


# Mating pool # adding the selected candidates from previous step
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


# In[24]:


#Crossover #ordered crossover
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child
def breedPopulation(matingpool, eliteSize):  # Elitism - to make sure that the fittest individuals are permitted to next gen properly
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children


# In[25]:


#Mutation #swap mutation
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual
def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


# In[26]:


#survivor selection # putting altogather the steps we have done before
def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    #print(len(selectionResults))
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    #print(len(children))
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


# In[27]:


#GA
def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print(pop)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
    
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
  
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute


# # Output

# In[39]:


# run
cityList = []

for i in range(0,25): # number of cities can be varied here.
    cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))

geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)


# In[ ]:





# In[29]:


# visualization
def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
    
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


# In[30]:


geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)


# # OUT2

# In[31]:


geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)


# In[32]:


geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)


# # OUT3

# In[35]:


geneticAlgorithm(population=cityList, popSize=100, eliteSize=15, mutationRate=0.01, generations=500)


# In[38]:


geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)


# In[ ]:






+

