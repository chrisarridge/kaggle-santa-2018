"""Code to interact with the problem on Kaggle and define the problem
"""

from typing import Callable
import numpy as np

import santaslittletsplib
import santaslittletsplib.constants
import santaslittletsplib.metrics


def sieve_of_eratosthenes(maximum_number: int, verbose=False) -> np.ndarray:
    """Find all prime numbers up to a maximum using the Sieve of Eratosthenes

    Parameters
    ----------
    maximum_number : int
        Maximum number to check for prime.
    verbose : bool, optional
        Display progress, by default False (No).

    Returns
    -------
    np.ndarray
        Array where 1 indicates that number is prime, 0 indicates it isn't prime.
    """
    # Store primes as an array where 0 is not prime and 1 is prime.
    prime = np.ones(maximum_number+1, dtype=np.uint8)
    prime[0] = 0
    prime[1] = 0

    for i in range(2, int(np.sqrt(maximum_number))+1):
        if verbose:
            print(i,np.linspace(0,maximum_number,maximum_number+1)[prime==1])
        prime[2*i:maximum_number+1:i] = 0

    if verbose:
        print('Complete: {}'.format(np.linspace(0,maximum_number,maximum_number+1)[prime==1]))

    return prime



def generate_simplified_problem(filename: str='data/cities.csv', xyscale: float=1.0,
                                prime_to_z: Callable[[np.ndarray],np.ndarray]=None) -> santaslittletsplib.Problem:
    """Generate a simplified problem that can be described in a TSP file.

    The code will optionally generate an additional z coordinate which depends on whether the city is prime or not.

    Parameters
    ----------
    filename : str, optional
        Filename of the data to load, default 'data/cities.csv'.
    xyscale : float, optional
        Scaling factor to apply to the x and y coordinates before turning into integers (as usual for a TSP file).
    prime_to_z : Callable[[np.ndarray],np.ndarray], optional
        Function which is used to create a z coordinate from whether the city is prime or not, by default None.

    Returns
    -------
    tsplib.Problem
        Problem object.
    """
    data = np.genfromtxt(filename, dtype=[('index','i'),('x','f8'),('y','f8')], delimiter=",", skip_header=1)

    problem = santaslittletsplib.Problem(santaslittletsplib.constants.ProblemType.TSP)
    problem.name = "Kaggle 2018 Santa TSP Challenge"
    problem.edge_weight_type = santaslittletsplib.constants.EdgeWeightType.EUCLIDEAN_2D
    problem._metric = santaslittletsplib.metrics.Euclidean()
    problem.dimension = len(data)

    # Mark all the prime city ids: prime=1 not prime=0.  Store as a custom variable
    # in the problem object.
    problem._prime = sieve_of_eratosthenes(problem.dimension-1)

    if prime_to_z is None:
        problem.node_coord_type = santaslittletsplib.constants.NodeCoordinateType.TWOD
        problem._nodes = np.zeros((len(data),2), dtype=np.int64)
    else:
        problem.node_coord_type = santaslittletsplib.constants.NodeCoordinateType.THREED
        problem._nodes = np.zeros((len(data),3), dtype=np.int64)

    # Complete node data.
    problem._nodes[data['index'],0] = (data['x']*xyscale).astype(np.int64)
    problem._nodes[data['index'],1] = (data['y']*xyscale).astype(np.int64)
    if prime_to_z is not None:
        problem._nodes[data['index'],2] = prime_to_z(problem._prime)

    return problem




def generate_full_problem(filename: str='data/cities.csv') -> santaslittletsplib.Problem:
    """Generate a full problem that cannot be described in a TSP file.

    The code will generate an additional z coordinate which depends on whether the city is prime or not,
    and a w coordinate coordinate which contains 0.0 if a city is prime and 0.1 if a city is not prime.

    Parameters
    ----------
    filename : str, optional
        Filename of the data to load, default 'data/cities.csv'.

    Returns
    -------
    tsplib.Problem
        Problem object.
    """
    data = np.genfromtxt(filename, dtype=[('index','i'),('x','f8'),('y','f8')], delimiter=",", skip_header=1)

    problem = santaslittletsplib.Problem(santaslittletsplib.constants.ProblemType.TSP)
    problem.name = "Kaggle 2018 Santa TSP Challenge"
    problem.edge_weight_type = santaslittletsplib.constants.EdgeWeightType.SPECIAL
    problem.node_coord_type = santaslittletsplib.constants.NodeCoordinateType.THREED
    problem._metric = SantaEuclidean()
    problem.dimension = len(data)

    # Mark all the prime city ids: prime=1 not prime=0 and store in the z coordinate
    problem._nodes = np.zeros((len(data),4), dtype=np.float64)
    problem._nodes[data['index'],0] = data['x']
    problem._nodes[data['index'],1] = data['y']
    problem._nodes[:,2] = sieve_of_eratosthenes(problem.dimension-1)
    problem._nodes[:,3] = 0.1
#    problem._nodes[problem._nodes[:,2]==1] = 0.0

    return problem



class SantaEuclidean(santaslittletsplib.metrics.BaseMetric):
    """Metric for the Kaggle 2018 Travelling Santa Problem.
    """

    def metric(self, tour_segment: np.ndarray, nodes:np.ndarray, start_step=0) -> np.ndarray:
        """Calculate the metric for the Kaggle 2018 Santa Challenge

        Every 10th edge is 10% longer *unless it comes from a CityId that is a prime
        number*.  In this problem the city ids are numbered from zero.  To calculate this
        we calculate the Euclidean distance using floating point numbers and then
        multiply every 10th step if it comes from a non-prime city id.

        Parameters
        ----------
        tour_segment : np.ndarray
            (m) Array containing the visited node indices (numbered from zero).
        nodes : np.ndarray
            (n x 2) or (n x 3) array of node coordinates.
        start_step : int, optional
            What step the tour_segment starts on, default zero.

        Returns
        -------
        np.ndarray
            (m-1) array containing the metric between each node.
        """
        # Get the vector along each edge and then the Euclidean distance along each edge.
        delta = nodes[tour_segment[1:],:2]-nodes[tour_segment[:-1],:2]
        edge_length = np.linalg.norm(delta,axis=1)

        # Get the number of each edge and then get the scaling.
        edge_number = np.arange(len(tour_segment)-1, dtype=np.int32) + start_step
        scale = 1.0 + nodes[tour_segment[:-1],3] * ((edge_number%10==0).astype(np.float64))
        edge_length *= scale
        return edge_length
