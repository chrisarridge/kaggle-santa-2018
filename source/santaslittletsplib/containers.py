"""Travelling Salesperson containers for TSPLIB tours and problem files.
"""
from typing import Union, List
import gzip

import numpy as np

from .metrics import BaseMetric, Euclidean, CeilEuclidean, Geographical, Manhattan, MaximumLinf, AttPseudoEuclidean
from .constants import ProblemType, EdgeWeightType, NodeCoordinateType, DisplayDataType
from .exceptions import MissingMetric

class Solution:
    """Solution container class
    """

    def __init__(self):
        """Initialise
        """
        self._comments = []
        self._name = None
        self._data = None
        self._dimension = None

    def __repr__(self) -> str:
        """Return a readable representation.

        Returns
        -------
        str
            Representation string.
        """
        return "Tour {} with {} vertices".format(self._name, self._dimension)

    def __str__(self) -> str:
        """Return a string representation of the solution.

        Returns
        -------
        str
            String representation.
        """
        return "Tour {} with {} vertices".format(self._name, self._dimension)

    def save(self, filename: str, force_gzip: bool=False):
        """Save the tour as a TSP tour file.

        Parameters
        ----------
        filename : str
            Filename to save the file as.  If it ends with gzip then the file will be gzip compressed.
        force_gzip : bool, optional
            if True, the filename will be gzipped even if the extension is not ".gz", by default False

        Raises
        ------
        ValueError
            If the solution doesn't have all the required information or is inconsistent.
        """

        # Fail if the solution doesn't have all the required information.
        if self._dimension is None:
            raise ValueError
        if self._data is None:
            raise ValueError
        if self._dimension != len(self._data):
            raise ValueError

        # Select the correct file handler: standard i/o or gzip.
        file_handler = open
        if filename[-3:]=='.gz' or force_gzip==True:
            file_handler = gzip.open

        # Write out the file.
        with file_handler(filename,'rt') as fh:
            if self._name is not None:
                fh.write('NAME : ' + self._name + '\n')
            [fh.write('COMMENT : ' + c + '\n') for c in self._comments]
            fh.write('DIMENSION : {}\n'.format(self._dimension))
            fh.write('TOUR_SECTION')
            [fh.write('{:i}\n'.format(x) for x in self._data)]
            fh.write('-1\nEOF\n')

    @property
    def name(self) -> str:
        """Return the name of the soltuion

        Returns
        -------
        str
            Name string.
        """
        return self._name

    @name.setter
    def name(self, v: str):
        """Set the name of the solution

        Parameters
        ----------
        v : str
            Name string.
        """
        self._name = v

    @property
    def comments(self) ->List[str]:
        """Get the list of comments.

        Returns
        -------
        List[str]
            Comments list.
        """
        return self._comments

    def append_comment(self, v: str):
        """Append a comment to the solution.

        Parameters
        ----------
        v : str
            Comment string.
        """
        self._comments.append(v)

    @property
    def dimension(self) -> int:
        """Get the dimension of the solution.

        Returns
        -------
        int
            Dimension.
        """
        return self._dimension

    @dimension.setter
    def dimension(self, v: int):
        """Set the dimension of the solution.

        Parameters
        ----------
        v : int
            Solution length.

        Raises
        ------
        TypeError
            If the dimension isn't an integer.
        """
        if not isinstance(v, int):
            raise TypeError
        else:
            self._dimension = v

    @property
    def data(self) -> np.ndarray:
        """Get access to the tour data.

        Returns
        -------
        np.ndarray
            Tour data.
        """
        return self._data

    @data.setter
    def data(self, v: np.ndarray):
        """Set the tour data.

        Parameters
        ----------
        v : np.ndarray
            Tour data to set.
        """
        self._data = v
        self._dimension = len(v)



class Problem:
    """TSP Problem container}
    """
    _problem_types_string_mapping = {ProblemType.TSP: "Symmetrical Travelling Salesperson",
                    ProblemType.ATSP: "Asymmetrical Travelling Salesperson",
                    ProblemType.SOP: "Sequential Ordering",
                    ProblemType.HCP: "Hamiltonian Cycle",
                    ProblemType.CVRP: "Capacitated Vehicle Routing"}

    def __init__(self, type : ProblemType):
        """Initialise

        Parameters
        ----------
        type : ProblemType
            Problem type.
        """
        self._name = "unnamed"
        self._comments = []
        self._type = type
        self._dimension = None
        self._edge_weight_type = None
        self._metric = None
        self._node_coord_type = NodeCoordinateType.NO_COORDINATES
        self._edge_weight_format = None
        self._display_data_type = DisplayDataType.COORD_DISPLAY

        self._edge_weights = None
        self._nodes = None
        self._display_data = None

    def __repr__(self) -> str:
        """Return representation of the problem.

        Returns
        -------
        str
            Representation of the problem.
        """
        return "{} Problem ({}) with {} nodes".format(self._problem_types_string_mapping[self._type], self._name, self._dimension)

    def __str__(self) -> str:
        """Return string representation of the problem.

        Returns
        -------
        str
            String representation of the problem.
        """
        return "{} Problem ({}) with {} nodes".format(self._problem_types_string_mapping[self._type], self._name, self._dimension)

    def cost(self, tour: Union[Solution,np.ndarray], metric: BaseMetric=None) -> Union[int,float]:
        """Get the cost function for this problem given a solution and optionally a metric.

        Solutions are automatically made symmetrical, e.g., if the tour has edges [0,1,2]
        then the cost of [0,1,2,0] will be computed.

        Parameters
        ----------
        tour : Union[Solution,np.ndarray]
            Solution to use to calculate the cost or a numpy array explicitly giving a tour.
        metric : Cost, optional
            Metric to use, this is ignored if we are using explicit weights, or if None
            we default to using the metric stored inside the problem.

        Returns
        -------
        Union[int,float]
            Cost.
        """

        # Get the tour data.
        if isinstance(tour,Solution):
            data = tour._data
        elif isinstance(tour,np.ndarray):
            data = tour

        # If we have no metric function, then just use the edge weight
        # matrix.
        if self._edge_weight_type==EdgeWeightType.EXPLICIT:
            cost = np.sum(self._edge_weights[data[1:],data[:-1]])
            if self._type==ProblemType.TSP:
                cost += self._edge_weights[data[-1],data[0]]

        else:
            # Make sure we have a metric we can use and store it
            # in a local temporary variable.
            if metric is None and self._metric is None:
                raise MissingMetric()

            this_metric = self._metric
            if metric is not None:
                this_metric = metric

            # Compute the cost.
            cost = this_metric(data, self._nodes)

            # If the problem is symmetrical then add the cost of
            # the last edge to go back to the beginning.
            if self._type==ProblemType.TSP:
                cost += this_metric([data[-1],data[0]], self._nodes)

        return cost

    @property
    def name(self) -> str:
        """Return the name of the soltuion

        Returns
        -------
        str
            Name string.
        """
        return self._name

    @name.setter
    def name(self, v: str):
        """Set the name of the solution

        Parameters
        ----------
        v : str
            Name string.
        """
        self._name = v

    @property
    def comments(self) ->List[str]:
        """Get the list of comments.

        Returns
        -------
        List[str]
            Comments list.
        """
        return self._comments

    def append_comment(self, v: str):
        """Append a comment to the solution.

        Parameters
        ----------
        v : str
            Comment string.
        """
        self._comments.append(v)

    @property
    def dimension(self) -> int:
        """Get the dimension of the solution.

        Returns
        -------
        int
            Dimension.
        """
        return self._dimension

    @dimension.setter
    def dimension(self, v: int):
        """Set the dimension of the solution.

        Parameters
        ----------
        v : int
            Solution length.

        Raises
        ------
        TypeError
            If the dimension isn't an integer.
        """
        if not isinstance(v, int):
            raise TypeError
        else:
            self._dimension = v

    @property
    def edge_weight_type(self) -> EdgeWeightType:
        """Get the edge weight type.

        Returns
        -------
        EdgeWeightType
            The problem edge weight type.
        """
        return self._edge_weight_type

    @edge_weight_type.setter
    def edge_weight_type(self, v: EdgeWeightType):
        """Set the edge weight type

        Parameters
        ----------
        v : EdgeWeightType
            New edge weight type for the problem.

        Raises
        ------
        TypeError
            If the new type is not an EdgeWeightType
        """
        if not isinstance(v, EdgeWeightType):
            raise TypeError
        self._edge_weight_type = v

        if self._edge_weight_type==EdgeWeightType.ATT:
            self._metric = AttPseudoEuclidean()
        elif self._edge_weight_type==EdgeWeightType.CEILING_2D:
            self._metric = CeilEuclidean()
        elif self._edge_weight_type==EdgeWeightType.EUCLIDEAN_2D:
            self._metric = Euclidean()
        elif self._edge_weight_type==EdgeWeightType.EUCLIDEAN_3D:
            self._metric = Euclidean()
        elif self._edge_weight_type==EdgeWeightType.GEOGRAPHICAL:
            self._metric = Geographical()
        elif self._edge_weight_type==EdgeWeightType.MANHATTAN_2D:
            self._metric = Manhattan()
        elif self._edge_weight_type==EdgeWeightType.MANHATTAN_3D:
            self._metric = Manhattan()
        elif self._edge_weight_type==EdgeWeightType.MAXIMUM_2D:
            self._metric = MaximumLinf()
        elif self._edge_weight_type==EdgeWeightType.MAXIMUM_3D:
            self._metric = MaximumLinf()
        else:
            self._metric = None

    @property
    def node_coord_type(self) -> NodeCoordinateType:
        """Get the problem node coordinate type.

        Returns
        -------
        NodeCoordinateType
            Problem node coordinate type.
        """
        return self._node_coord_type

    @node_coord_type.setter
    def node_coord_type(self, v: NodeCoordinateType):
        """Set the problem node coordinate type.

        Parameters
        ----------
        v : NodeCoordinateType
            New problem node coordinate type.

        Raises
        ------
        TypeError
            If the new type is not a NodeCoordinateType.
        """
        if not isinstance(v, NodeCoordinateType):
            raise TypeError
        self._node_coord_type = v        

    @property
    def nodes(self) -> np.ndarray:
        return self._nodes
