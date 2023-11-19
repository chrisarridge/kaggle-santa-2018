"""Types and constants
"""
import enum

class ProblemType(enum.Enum):
    """TSPLIB File type

    Attributes
    ----------
    TSP
        Symmetrical travelling salesperson problem
    ATSP
        Asymmetrical travelling salesperson problem
    SOP
        Sequential ordering problem
    HCP
        Hamiltonian cycle problem
    CVRP
        Capacitated vehicle routing problem
    """
    TSP = "TSP"
    ATSP = "ATSP"
    SOP = "SOP"
    HCP = "HCP"
    CVRP = "CVRP"



class EdgeWeightType(enum.Enum):
    """Edge weight types

    Attributes
    ----------
    EXPLICIT
        Weights are explicitly provided in an EDGE_WEIGHT_SECTION
    EUCLIDEAN_2D
        Euclidean distances in 2D.
    EUCLIDEAN_3D
        Euclidean distances in 3D.
    MAXIMUM_2D
        Maximum distances in 2D.
    MAXIMUM_3D
        Maximum distances in 3D.
    MANHATTAN_2D
        Manhattan distances in 2D.
    MANHATTAN_3D
        Manhattan distances in 3D.
    CEILING_2D
        Euclidean distances in 2D but rounded up to a whole number.
    GEOGRAPHICAL
        Geographical distances.
    ATT
        Special weighting function for att48 and att532 problems.
    XRAY1
        Special function for crystallography problems (version 1).
    XRAY2
        Special function for crystallography problems (version 2).
    SPECIAL
        Another special function documented elsewhere.
    """
    EXPLICIT = "EXPLICIT"
    EUCLIDEAN_2D = "EUC_2D"
    EUCLIDEAN_3D = "EUC_3D"
    MAXIMUM_2D = "MAX_2D"
    MAXIMUM_3D = "MAX_3D"
    MANHATTAN_2D = "MAN_2D"
    MANHATTAN_3D = "MAN_3D"
    CEILING_2D = "CEIL_2D"
    GEOGRAPHICAL = "GEO"
    ATT = "ATT"
    XRAY1 = "XRAY1"
    XRAY2 = "XRAY2"
    SPECIAL = "SPECIAL"



class EdgeWeightFormat(enum.Enum):
    """Edge weight format

    Attributes
    ----------
    FUNCTION
        Edge weights are specified by a function in EdgeWeightType
    FULL_MATRIX
        A full m x m matrix of edge weights between each node.
    UPPER_ROW
        Upper triangular matrix without diagonal (zero) entries arranged row-wise.
    LOWER_ROW
        Lower triangular matrix without diagonal (zero) entries arranged row-wise.
    UPPER_DIAG_ROW
        Upper triangular matrix with diagonal (zero) entries arranged row-wise.
    LOWER_DIAG_ROW
        Lower triangular matrix with diagonal (zero) entries arranged row-wise.
    UPPER_COL
        Upper triangular matrix without diagonal (zero) entries arranged column-wise.
    LOWER_COL
        Lower triangular matrix without diagonal (zero) entries arranged column-wise.
    UPPER_DIAG_COL
        Upper triangular matrix with diagonal (zero) entries arranged column-wise.
    LOWER_DIAG_COL
        Lower triangular matrix with diagonal (zero) entries arranged column-wise.
    """
    FUNCTION = "FUNCTION"
    FULL_MATRIX = "FULL_MATRIX"
    UPPER_ROW = "UPPER_ROW"
    LOWER_ROW = "LOWER_ROW"
    UPPER_DIAG_ROW = "UPPER_DIAG_ROW"
    LOWER_DIAG_ROW = "LOWER_DIAG_ROW"
    UPPER_COL = "UPPER_COL"
    LOWER_COL = "LOWER_COL"
    UPPER_DIAG_COL = "UPPER_DIAG_COL"
    LOWER_DIAG_COL = "LOWER_DIAG_COL"



class EdgeDataFormat(enum.Enum):
    """Edge data formats

    Attributes
    ----------
    EDGE_LIST
        The graph is provided as a list of edges.
    ADJ_LIST
        The graph is provided by a list of adjaency.
    """
    EDGE_LIST = "EDGE_LIST"
    ADJ_LIST = "ADJ_LIST"



class NodeCoordinateType(enum.Enum):
    """Node coordinate types

    Attributes
    ----------
    TWOD
        Two dimensional coordinates.
    THREED
        Three dimensional coordinates.
    FOURD
        Four dimensional coordinates - not supported by TSP Standard.
    NO_COORDINATES
        No coordinates are given.
    """
    TWOD = "TWOD_COORDS"
    THREED = "THREED_COORDS"
    FOURD = "FOURD_COORDS"
    NO_COORDINATES = "NO_COORDS"



class DisplayDataType(enum.Enum):
    """How to display the nodes

    Attributes
    ----------
    COORD_DISPLAY
        Draw based on actual node coordinates.
    TWOD_DISPLAY
        Explicit 2D coordinates are given for each node.
    NO_DISPLAY
        It's not possible to draw this problem.
    """
    COORD_DISPLAY = "COORD_DISPLAY"
    TWOD_DISPLAY = "TWOD_DISPLAY"
    NO_DISPLAY = "NO_DISPLAY"
