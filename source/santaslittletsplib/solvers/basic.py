"""Simple solvers.
"""
import itertools
import math

import numpy as np

from ..containers import *
from ..constants import *


def nearest_neighbour(problem: Problem) -> Solution:
    """Find a solution to a TSP problem using nearest-neighbours.

    Parameters
    ----------
    problem : tsplib.Problem
        Problem to solve.

    Returns
    -------
    tsplib.Solution
        Solution we found.
    """

    remaining_cities = list(range(problem.dimension))
    tour = np.zeros(problem.dimension, dtype=np.int64)-1

    # Tour starts with city 0 so set that and remove it from the remaining city list.
    tour[0] = remaining_cities[0]
    remaining_cities.remove(0)

    # For each city we compute the distance to all the remaining cities and choose the smallest.
    for indx in range(1,problem.dimension):
        tmp = [problem._metric([tour[indx-1],city], problem._nodes) for city in remaining_cities]
        min_city = np.argmin(tmp)
        tour[indx] = remaining_cities[min_city]
        remaining_cities.remove(remaining_cities[min_city])

    # Having found the nearest neighbour solution, construct a Solution object.
    sol = Solution()
    sol.name = "nearest-neighbour"
    sol.append_comment("Found using nearest-neighbour search")
    sol.append_comment("cost={}".format(problem.cost(tour)))
    sol.dimension = problem.dimension
    sol.data = tour

    return sol



def brute_force(problem: Problem) -> Solution:
    """Find solution to a TSP problem using a brute-force search of all permutations.

    Parameters
    ----------
    problem : tsplib.Problem
        Problem to solve.

    Returns
    -------
    tsplib.Solution
        Optimal solution.
    """

    # Generate permutations that all start with node 0.
    num_perms = math.factorial(problem.dimension-1)
    perms = np.zeros((num_perms, problem.dimension), dtype=np.int64)
    perms[:,1:] = np.array(list(itertools.permutations(np.arange(1,problem.dimension))))

    # For each permutation calculate the cost
    costs = [problem.cost(perms[i,:]) for i in range(perms.shape[0])]
    pp = np.argsort(costs)

    # Having found the optimal solution, construct a Solution object.
    tour = Solution()
    tour.name = "brute-force"
    tour.append_comment("Found using brute force search of permutations")
    tour.append_comment("cost={}".format(costs[pp[0]]))
    tour.dimension = problem.dimension
    tour.data = perms[pp[0]]

    return tour

