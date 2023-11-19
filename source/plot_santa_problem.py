"""Plot the Kaggle Santa Prime Paths problem space"""

import matplotlib.pyplot as plt
import numpy as np

import kagglesanta

def main():
    problem = kagglesanta.generate_full_problem()

    # Get a colour map so we can colour the prime cityId's separately,
    # also scale the prime cityId's so they are larger.
    cmap = plt.get_cmap("tab10")
    colours = [cmap(int(p)+2) for p in problem.nodes[:,2]]
    marker_sizes = [1+1*p for p in problem.nodes[:,2]]

    # The combination of marker size, figure size, and DPI is selected to make each
    # city be a few pixels across.
    plt.figure(figsize=(np.max(problem.nodes[:,0])/500,np.max(problem.nodes[:,1])/500))
    plt.scatter(problem.nodes[:,0], problem.nodes[:,1], s=marker_sizes, c=colours, marker='o', edgecolors="none")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal")
    plt.savefig("santa-problem-space.png", dpi=150)
    plt.close()


if __name__=="__main__":
    main()