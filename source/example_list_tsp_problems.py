"""Example listing all the TSP problems"""
import glob

import numpy as np

import santaslittletsplib

def main():
    files = glob.glob("data/tsplib95/tsp/*.tsp.gz")

    print("name, problem_type, dimension, edge_weight_type, node_coord_type")
    for filename in files:
        try:
            problem = santaslittletsplib.load(filename)
            print('{}, {}, {}, {}, {}, {}'.format(filename, problem.name, problem._type, problem.dimension, problem.edge_weight_type, problem.node_coord_type))
        except santaslittletsplib.BadTspFile as e:
            print(filename, e)


if __name__=="__main__":
    main()