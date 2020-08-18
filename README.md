# TSP-VRP-_-Travelling-SalesMan-Problem-_-solution-using-Genetic-Algorithm
# Vehicle Routing Problem
The vehicle routing problem (VRP) is a combinatorial optimization and integer programming problem seeking to service a number of customers with a fleet of vehicles. 
VRP is an important problem in the fields of transportation, distribution and logistics. 
Often the context is that of delivering goods located at a central depot to customers who have placed orders for such goods. 
Implicit is the goal of minimizing the cost of distributing the goods. Many methods have been developed for searching for good solutions to the problem, but finding global minimum for the cost function is computationally complex.

Design and develop a genotype model for representing VRP, and also develop an Evolutionary Algorithm that can evolve candidate solutions and find the solution.

We are considering the cost function as the distance covered to travel through all the cities exactly once.

We are reducing the the VRP to travelling salesman problem by the assumption that

If we can reduce the distance travelled to cover the longest route, then we can then we can complete the each route with mininal time and there by accomplish delivery of 
larger number of orders by allocating multiple routes to vehicle which completes the delivery fast and there by we can reduce the cost and maximise the profit.


# Travelling salesman Problem

## Constraints

Each city needs to be visited exactly one time
We must return to the starting city, so our total distance needs to be calculated accordingly

## opimisation problem

Our objective is to minimize the distance travelled by saleseman(here vehicle) to travell through all the cities.

# Design of Experiments

## GA

Representation -------- Permutation of Integer tuples which represents the city

Fitness -------- Inverse of the route distance

Recombination -------- ordered crossover, elitism size 20

Mutation -------- Swap

Mutation Probability -------- 0.01

Parent Selection -------- Best 2

Survival Selection -------- rank, roulette wheel, elitislm 20

Population size -------- 100

Number of Offspring -------- 2

Initialization -------- Random

Termination condition -------- 500 generations
