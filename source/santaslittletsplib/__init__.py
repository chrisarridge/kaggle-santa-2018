"""TSPLIB parsing and tools to manipulate tours.
"""

__version__ = '0.1'
__author__ = 'Chris Arridge'

from .exceptions import BadTspFile, ParsingError, UnsupportedEdgeWeight, UnsupportedProblemType
from .libparser import load
from .containers import Problem, Solution
from .constants import *