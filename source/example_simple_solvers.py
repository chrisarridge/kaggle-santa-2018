"""Example of using brute force and nearest neighbour solvers"""
import numpy as np

import santaslittletsplib
import santaslittletsplib.solvers

def main():
    # Setup a simple symmetrical TSP problem with 6 vertices.
    prob = santaslittletsplib.Problem(santaslittletsplib.ProblemType.TSP)
    prob.dimension = 6
    prob._nodes = np.zeros((6,2), dtype=np.int64)
    prob._nodes[0,:] = [11004,42103]
    prob._nodes[1,:] = [11417, 42983]
    prob._nodes[2,:] = [11522, 42842]
    prob._nodes[3,:] = [11751, 42814]
    prob._nodes[4,:] = [12058, 42196]
    prob._nodes[5,:] = [12387, 43335]
    prob.edge_weight_type = santaslittletsplib.EdgeWeightType.EUCLIDEAN_2D

    # Find a tour using a brute force search.
    tour = santaslittletsplib.solvers.brute_force(prob)
    print(tour.comments)
    print(prob.cost(tour))
    print(tour.data)

    # Find a tour using a nearest neighbour search.
    tour = santaslittletsplib.solvers.nearest_neighbour(prob)
    print(tour.comments)
    print(prob.cost(tour))
    print(tour.data)


if __name__=="__main__":
    main()