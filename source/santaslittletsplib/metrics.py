"""Metric classes
"""
from typing import Union
import numpy as np


def nint(x: np.ndarray, inttype: np.number=np.int64) -> np.ndarray:
    """Compute the nearest integer as defined in TSPLIB95

    Parameters
    ----------
    x : np.ndarray
        The NumPy array to compute the nearest integer on.
    inttype : np.number, optional
        The integer type we will return, by default np.int64

    Returns
    -------
    np.ndarray
        Returned array of nearest integers.
    """
    return (x+0.5).astype(inttype)



class BaseMetric:
    """Metric and cost function base class
    """
    def __call__(self, tour_segment: np.ndarray, nodes:np.ndarray) -> Union[int,float]:
        """Computes the total cost function by summing up all the metric calculations between vertices

        Parameters
        ----------
        tour_segment : np.ndarray
            (m) Array containing the visited node indices (numbered from zero).
        nodes : np.ndarray
            (n x 2) or (n x 3) array of node coordinates.

        Returns
        -------
        Union[int,float]
            Integer or floating point cost function.
        """
        return np.sum(self.metric(tour_segment, nodes))

    def metric(self, tour_segment: np.ndarray, nodes:np.ndarray) -> np.ndarray:
        """Calculate the metric - abstract function

        Subclasses should write the appropriate metric function to return an array
        with each entry giving the metric for that edge.
        
        Parameters
        ----------
        tour_segment : np.ndarray
            (m) Array containing the visited node indices (numbered from zero).
        nodes : np.ndarray
            (n x 2) or (n x 3) array of node coordinates.

        Returns
        -------
        np.ndarray
            (m-1) array containing the metric between each node.
        """
        pass



class Euclidean(BaseMetric):
    """Euclidean (2D or 3D) L2 metric strictly as defined for TSPLIB95

    """
    def metric(self, tour_segment: np.ndarray, nodes:np.ndarray) -> np.ndarray:
        """Calculate Euclidean metric as defined in TSPLIB95

        This function calculates the L2 metric in two- or
        three-dimensions and returns the metric for each edge.

        Parameters
        ----------
        tour_segment : np.ndarray
            (m) Array containing the visited node indices (numbered from zero).
        nodes : np.ndarray
            (n x 2) or (n x 3) array of node coordinates.

        Returns
        -------
        np.ndarray
            (m-1) array containing the metric between each node.
        """
        delta = nodes[tour_segment[1:],:]-nodes[tour_segment[:-1],:]
        return nint(np.linalg.norm(delta,axis=1))



class AttPseudoEuclidean(BaseMetric):
    """ATT Pseudo-Euclidean metric in 2D as defined in TSPLIB95.

    """
    def metric(self, tour_segment: np.ndarray, nodes:np.ndarray) -> np.ndarray:
        """Calculate ATT metric as defined in TSPLIB95

        Parameters
        ----------
        tour_segment : np.ndarray
            (m) Array containing the visited node indices (numbered from zero).
        nodes : np.ndarray
            (n x 2) or (n x 3) array of node coordinates.

        Returns
        -------
        np.ndarray
            (m-1) array containing the metric between each node.
        """
        delta = nodes[tour_segment[1:],:]-nodes[tour_segment[:-1],:]
        rij = np.linalg.norm(delta, axis=1)/np.sqrt(10.0)
        tij = nint(rij)
        dij = tij.copy()
        dij[tij<rij] = tij[tij<rij] + 1
        return dij


class CeilEuclidean(BaseMetric):
    """Euclidean L2 metric rounded up to the next integer.

    """
    def metric(self, tour_segment: np.ndarray, nodes:np.ndarray) -> np.ndarray:
        """Euclidean L2 metric rounded up to the next integer.

        Parameters
        ----------
        tour_segment : np.ndarray
            (m) Array containing the visited node indices (numbered from zero).
        nodes : np.ndarray
            (n x 2) or (n x 3) array of node coordinates.

        Returns
        -------
        np.ndarray
            (m-1) array containing the metric between each node.
        """
        delta = nodes[tour_segment[1:],:]-nodes[tour_segment[:-1],:]
        return np.ceil(np.linalg.norm(delta,axis=1)).astype(np.int64)



class Geographical(BaseMetric):
    """Geographical metric as defined for TSPLIB95
    """
    def __init__(self, radius: float=6378.388):
        """Initialise

        Parameters
        ----------
        radius : float, optional
            Radius of the body on which to calculate distances, by default 6378.388 km.
        """
        self._radius = radius

    def metric(self, tour_segment: np.ndarray, nodes:np.ndarray) -> np.ndarray:
        """Geographical metric as defined for TSPLIB95

        Parameters
        ----------
        tour_segment : np.ndarray
            (m) Array containing the visited node indices (numbered from zero).
        nodes : np.ndarray
            (n x 2) or (n x 3) array of node coordinates.

        Returns
        -------
        np.ndarray
            (m-1) array containing the metric between each node.
        """

        # extract degrees and minutes and calculate latitude and longitude.
        latitude_degrees = nodes[:,0].astype(np.int64)
        latitude_minutes = nodes[:,0] - latitude_degrees
        latitude = np.radians(latitude_degrees + latitude_minutes*5.0/3.0)
        longitude_degrees = nodes[:,1].astype(np.int64)
        longitude_minutes = nodes[:,1] - longitude_degrees
        longitude = np.radians(longitude_degrees + longitude_minutes*5.0/3.0)

        q1 = np.cos(longitude[tour_segment[:-1]] - longitude[tour_segment[1:]])
        q2 = np.cos(latitude[tour_segment[:-1]] - latitude[tour_segment[1:]])
        q3 = np.cos(latitude[tour_segment[:-1]] + latitude[tour_segment[1:]])
        delta = (self._radius*np.arccos( 0.5*((1.0+q1)*q2 - (1.0-q1)*q3) ) + 1.0).astype(np.int64)

        return delta





class FloatEuclidean(BaseMetric):
    """Euclidean (2D or 3D) L2 metric as a floating point number which differs from the TSPLIB95 definition.

    """
    def metric(self, tour_segment: np.ndarray, nodes:np.ndarray) -> np.ndarray:
        """Calculate Euclidean metric

        This function calculates the L2 metric in two- or
        three-dimensions and returns the metric for each edge.

        Parameters
        ----------
        tour_segment : np.ndarray
            (m) Array containing the visited node indices (numbered from zero).
        nodes : np.ndarray
            (n x 2) or (n x 3) array of node coordinates.

        Returns
        -------
        np.ndarray
            (m-1) array containing the metric between each node.
        """
        delta = nodes[tour_segment[1:],:]-nodes[tour_segment[:-1],:]
        return np.linalg.norm(delta,axis=1)



class Manhattan(BaseMetric):
    """Manhattan (2D or 3D) L1-metric strictly as defined for TSPLIB95

    """
    def metric(self, tour_segment: np.ndarray, nodes:np.ndarray) -> np.ndarray:
        """Calculate Manhattan L1 metric as defined in TSPLIB95

        This function calculates the L1 metric in two- or
        three-dimensions and returns the metric for each edge.

        Parameters
        ----------
        tour_segment : np.ndarray
            (m) Array containing the visited node indices (numbered from zero).
        nodes : np.ndarray
            (n x 2) or (n x 3) array of node coordinates.

        Returns
        -------
        np.ndarray
            (m-1) array containing the metric between each node.
        """
        delta = np.abs(nodes[tour_segment[1:],:]-nodes[tour_segment[:-1],:])
        return nint(np.sum(delta,axis=1))



class MaximumLinf(BaseMetric):
    """Maximum (2D or 3D) distance L_\inf metric strictly as defined for TSPLIB95

    """
    def metric(self, tour_segment: np.ndarray, nodes:np.ndarray) -> np.ndarray:
        """Calculate Maximum L_\inf metric as defined in TSPLIB95

        This function calculates the L_\inf metric in two- or
        three-dimensions and returns the metric for each edge.

        Parameters
        ----------
        tour_segment : np.ndarray
            (m) Array containing the visited node indices (numbered from zero).
        nodes : np.ndarray
            (n x 2) or (n x 3) array of node coordinates.

        Returns
        -------
        np.ndarray
            (m-1) array containing the metric between each node.
        """
        delta = nint(np.abs(nodes[tour_segment[1:],:]-nodes[tour_segment[:-1],:]))
        return np.max(delta,axis=1)
