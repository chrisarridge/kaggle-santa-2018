"""Exception classes for parsing TSPLIB files
"""

class BadTspFile(Exception):
    """Raised if the TSP or TOUR file is badly formatted"""
    pass

class ParsingError(Exception):
    """Raised if the parser got confused when reading a TSP or TOUR file"""
    pass

class UnsupportedEdgeWeight(Exception):
    """Raised if the TSP file contained unsupported edge weight formats."""
    pass

class UnsupportedProblemType(Exception):
    """Raised if the TSP file contained an unsupported problem type"""
    pass

class MissingMetric(Exception):
    """Raised if a problem has no metric to compute a cost function with"""
    pass