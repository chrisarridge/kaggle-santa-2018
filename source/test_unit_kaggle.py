"""Unit test the Kaggle Santa problem-specific code for prime numbers and metrics"""
from typing import Union
import unittest

import numpy as np

import kagglesanta


class TestSieve(unittest.TestCase):
    """Test case to check we calculate primes correctly"""

    def test_sequence(self):
        """Test a whole sequence of primes up to and including 541
        """
        full_list = np.array([2,3,5,7,11,13,17,19,23,29,31,37,
                        41,43,47,53,59,61,67,71,73,79,83,89,
                        97,101,103,107,109,113,127,131,
                        137,139,149,151,157,163,167,173,179,
                        181,191,193,197,199,211,223,227,
                        229,233,239,241,251,257,263,269,
                        271,277,281,283,293,307,311,313,
                        317,331,337,347,349,353,359,367,
                        373,379,383,389,397,401,409,419,
                        421,431,433,439,443,449,457,461,
                        463,467,479,487,491,499,503,509,
                        521,523,541])
        for max_number in range(30,541+1):
            answer = self.array_from_primes(max_number, full_list[full_list<=max_number])
            primes = kagglesanta.sieve_of_eratosthenes(max_number)
            self.assertCountEqual(primes, answer)
            self.assertTrue(np.array_equal(primes, answer))
#            print('{}: primes={}'.format(max_number, self.primes_from_array(max_number, primes)))

    @staticmethod
    def array_from_primes(max_x: int, x: Union[list,np.ndarray,tuple]):
        """Method to generate an array where every entry is one if the number is a prime.

        Given x=[2,3,5] and max_x=5, this will generate an array with six entries where the
        2nd, 3rd, and 5th entries are one and the rest are zeros:  [0, 0, 1, 1, 0, 1]. 

        Parameters
        ----------
        max_x : int
            Maximum value to generate (will produce an array of length max_x+1).
        x : Union[list,np.ndarray,tuple]
            Numbers that are prime.

        Returns
        -------
        np.ndarray
        """
        answer = np.zeros(max_x+1,dtype=np.uint8)
        answer[x]=1
        return answer

    @staticmethod
    def primes_from_array(max_x: int, p: Union[list,np.ndarray,tuple]):
        """Given an array where primes are indicated by 1 this generates a list of the prime numbers.

        Parameters
        ----------
        p : Union[list,np.ndarray,tuple]
            Array where an entry is 1 if that index is prime, e.g., p=[0,0,1,1,0] indicates
            that 0 is not prime p[0]=0, but 2 is prim since p[2]=1.

        Returns
        -------
        np.ndarray
        """
        return np.linspace(0,len(p),len(p)+1)[p==1]



class TestMetric(unittest.TestCase):
    """Test case to make sure the Santa prime metric works correctly.
    """
    def test(self):

        # generate faked data - all the cities are randomly located.
        np.random.seed(1524)
        dimension = 14
        nodes = np.zeros((dimension,4), dtype=np.float64)
        nodes[:,0] = np.random.random(dimension)
        nodes[:,1] = np.random.random(dimension)
        nodes[:,2] = kagglesanta.sieve_of_eratosthenes(dimension-1)
        nodes[nodes[:,2]==0,3] = 0.1

        # get santa metric
        santa_metric = kagglesanta.SantaEuclidean()

        # tour where the 10th step comes from a prime city id
        # the 10th step is 11-12 where 11 is prime and so that
        # edge will not be 10% longer.
        tour = np.array([2,0,1,3,4,5,6,7,8,9,11,12,13], np.uint32)

        # try a standard Euclidean metric, this should be the same as
        # the santa metric.
        delta = nodes[tour[1:],:2]-nodes[tour[:-1],:2]
        euclidean_edges = np.linalg.norm(delta,axis=1)
        euclidean_cost = np.sum(euclidean_edges)
        self.assertAlmostEqual(euclidean_cost, santa_metric(tour, nodes))

        santa_edges = santa_metric.metric(tour, nodes)
        self.assertTrue(np.array_equal(euclidean_edges, santa_edges))

        # now try a tour where the 10th step comes from a non-prime city id
        # the 10th step is 10-12 where 10 is not prime, and so that
        # edge will be 10% longer.
        tour = np.array([2,0,1,3,4,5,6,7,8,9,10,12,13], np.uint32)
        delta = nodes[tour[1:],:2]-nodes[tour[:-1],:2]
        euclidean_edges = np.linalg.norm(delta,axis=1)
        euclidean_cost = np.sum(euclidean_edges)
        self.assertNotAlmostEqual(euclidean_cost, santa_metric(tour, nodes))

        santa_edges = santa_metric.metric(tour, nodes)
        self.assertFalse(np.array_equal(euclidean_edges, santa_edges))

if __name__=="__main__":
    unittest.main()
