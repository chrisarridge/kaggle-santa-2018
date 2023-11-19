"""Parser, file loader and container factory.
"""
from typing import Tuple, Union
import re
import gzip

import numpy as np

from .exceptions import BadTspFile, ParsingError, UnsupportedEdgeWeight
from .constants import ProblemType, EdgeWeightType, EdgeWeightFormat, EdgeDataFormat, NodeCoordinateType, DisplayDataType
from .containers import Solution, Problem
from .metrics import Euclidean, AttPseudoEuclidean, CeilEuclidean, Geographical, Manhattan, MaximumLinf



def parse_to_dict(filename: str, force_gzip: bool=False) -> Tuple[dict,dict]:
    """Load a TSPLIB file and construct a Problem or Tour object.

    Parameters
    ----------
    filename : str
        Filename to load.  If the filename ends with .gz then we will assume it's GZIP'd.  This can
        be overridden with the force_gzip option.
    force_gzip : bool, optional
        If True, then the reader will assume the file is gzipped, otherwise it will try to figure it
        out for itself.  Optional, default is False.

    Returns
    -------
    Tuple[dict,dict]
        Tuple containing the specification dictionary and the data dictionary.

    Raises
    ------
    BadTspFile
        If there is an internal problem with the TSPLIB file.
    ParsingError
        If the parser has become lost and cannot continue parsing.
    """
    # Dictionaries to store the return data.
    specification = {'comments':[]}
    data = {}

    # These enumeration mappings simplify our parsing by searching for TSPLIB file strings
    # and matching them to the appropriate enum.
    _edge_weight_type_map = {_x.value:_x for _x in EdgeWeightType}
    _problem_type_map = {_x.value:_x for _x in ProblemType}
    _edge_weight_format_map = {_x.value:_x for _x in EdgeWeightFormat}

    # Select the correct file handler: standard i/o or gzip.
    file_handler = open
    if filename[-3:]=='.gz' or force_gzip==True:
        file_handler = gzip.open

    # Process each line in the file.  This operates as a state machine and defaults to
    # searching for specification data.
    with file_handler(filename,'rt') as fh:
        state = 'specification'

        for _buffer in fh:
            buffer = _buffer.strip()

            if len(buffer)>0:
                # If we are in a specification state then search for
                # a specification string.
                if state=='specification':

                    g = re.search('(\w+)\s*:\s*([\s\S]+)',buffer)
                    if g is not None:
                        token = g.group(1)
                        content = g.group(2).strip()

                        if token=='NAME':
                            specification['name'] = content

                        elif token=='TYPE':
                            if content=='TOUR':
                                specification['type'] = 'TOUR'
                            else:
                                if content in _problem_type_map:
                                    specification['type'] = _problem_type_map[content]
                                else:
                                    raise BadTspFile('Unknown type of TSPLIB file <{}>'.format(content))

                        elif token=='COMMENT':
                            specification['comments'].append(content)

                        elif token=='CAPACITY':
                            specification['capacity'] = float(content)

                        elif token=='DIMENSION':
                            specification['dimension'] = int(content)

                        elif token=='EDGE_WEIGHT_TYPE':
                            if content in _edge_weight_type_map:
                                specification['edge_weight_type'] = _edge_weight_type_map[content]
                            else:
                                raise BadTspFile('Unknown edge weight type in TSPLIB file <{}>'.format(content))

                        elif token=='EDGE_WEIGHT_FORMAT':
                            if content in _edge_weight_format_map:
                                specification['edge_weight_format'] = _edge_weight_format_map[content]
                            else:
                                raise BadTspFile('Unknown edge weight format in TSPLIB file <{}>'.format(content))

                        elif token=='EDGE_DATA_FORMAT':
                            if content=='EDGE_LIST':
                                specification['edge_data_format'] = EdgeDataFormat.EDGE_LIST
                            elif content=='ADJ_LIST':
                                specification['edge_data_format'] = EdgeDataFormat.ADJ_LIST
                            else:
                                raise BadTspFile('Unknown edge data format in TSPLIB file <{}>'.format(content))

                        elif token=='NODE_COORD_TYPE':
                            if content=='TWOD_COORDS':
                                specification['node_coord_type'] = NodeCoordinateType.TWOD
                            elif content=='THREED_COORDS':
                                specification['node_coord_type'] = NodeCoordinateType.THREED
                            elif content=='NO_COORDS':
                                specification['node_coord_type'] = NodeCoordinateType.NO_COORDINATES
                            else:
                                raise BadTspFile('Unknown node coordinate type in TSPLIB file <{}>'.format(content))

                        elif token=='DISPLAY_DATA_TYPE':
                            if content=='COORD_DISPLAY':
                                specification['display_data_type'] = DisplayDataType.COORD_DISPLAY
                            elif content=='TWOD_DISPLAY':
                                specification['display_data_type'] = DisplayDataType.TWOD_DISPLAY
                            elif content=='NO_DISPLAY':
                                specification['display_data_type'] = DisplayDataType.NO_DISPLAY
                            else:
                                raise BadTspFile('Unknown display data type in TSPLIB file <{}>'.format(content))

                        else:
                            raise BadTspFile('Unknown keyword in TSPLIB file <{}>'.format(token))

                    elif buffer=='NODE_COORD_SECTION':
                        state = 'node_coords_data'
                        data['node_coords'] = {}

                    elif buffer=='DEPOT_SECTION':
                        state = 'depot_node_data'
                        data['depot_node'] = []

                    elif buffer=='DEMAND_SECTION':
                        state = 'demand_data'
                        data['demand'] = {}

                    elif buffer=='EDGE_DATA_SECTION':
                        state = 'edge_data'
                        data['edge'] = {}

                    elif buffer=='FIXED_EDGES_SECTION':
                        state = 'fixed_edge_data'
                        data['fixed_edges'] = {}

                    elif buffer=='DISPLAY_DATA_SECTION':
                        state = 'display_data'
                        data['display'] = {}

                    elif buffer=='TOUR_SECTION':
                        state = 'tour_data'
                        data['tour'] = []

                    elif buffer=='EDGE_WEIGHT_SECTION':
                        state = 'edge_weight_data'
                        data['edge_weights'] = []

                    elif buffer=='EOF':
                        state='done'

                elif (state=='node_coords_data' or state=='depot_node_data' or state=='demand_data' or state=='edge_data'
                    or state=='fixed_edge_data' or state=='display_data' or state=='tour_data' or state=='edge_weight_data'):

                    if buffer=='NODE_COORD_SECTION':
                        if 'node_coords' in data:
                            raise BadTspFile('Duplicate node coordinate data in TSPLIB file')
                        state = 'node_coords_data'
                        data['node_coords'] = {}

                    elif buffer=='DEPOT_SECTION':
                        if 'depot_node' in data:
                            raise BadTspFile('Duplicate debot node data in TSPLIB file')
                        state = 'depot_node_data'
                        data['depot_node'] = []

                    elif buffer=='DEMAND_SECTION':
                        if 'demand' in data:
                            raise BadTspFile('Duplicate demand data in TSPLIB file')
                        state = 'demand_data'
                        data['demand'] = {}

                    elif buffer=='EDGE_DATA_SECTION':
                        if 'edge' in data:
                            raise BadTspFile('Duplicate edge data in TSPLIB file')
                        state = 'edge_data'
                        data['edge'] = {}

                    elif buffer=='FIXED_EDGES_SECTION':
                        if 'fixed_edges' in data:
                            raise BadTspFile('Duplicate fixed edge data in TSPLIB file')
                        state = 'fixed_edge_data'
                        data['fixed_edges'] = {}

                    elif buffer=='DISPLAY_DATA_SECTION':
                        if 'display' in data:
                            raise BadTspFile('Duplicate display data in TSPLIB file')
                        state = 'display_data'
                        data['display'] = {}

                    elif buffer=='TOUR_SECTION':
                        if 'tour' in data:
                            raise BadTspFile('Duplicate tour data in TSPLIB file')
                        state = 'tour_data'
                        data['tour'] = []

                    elif buffer=='EDGE_WEIGHT_SECTION':
                        if 'edge_weight' in data:
                            raise BadTspFile('Duplicate edge weight data in TSPLIB file')
                        state = 'edge_weight_data'
                        data['edge_weights'] = []

                    elif buffer=='EOF':
                        state = 'done'

                    else:
                        if buffer=='-1':
                            state = 'specification'

                        elif state=='node_coords_data':
                            if 'node_coord_type' not in specification:
                                specification['node_coord_type'] = NodeCoordinateType.TWOD
                            tmp = buffer.split()
                            if specification['node_coord_type']==NodeCoordinateType.TWOD:
                                data['node_coords'][int(tmp[0])] = [float(tmp[1]),float(tmp[2])]
                            elif specification['node_coord_type']==NodeCoordinateType.THREED:
                                data['node_coords'][int(tmp[0])] = [float(tmp[1]),float(tmp[2]),float(tmp[3])]
                            else:
                                raise BadTspFile('Cannot parse node coordinate data, unknown node coordinate type')

                        elif state=='depot_node_data':
                            data['depot_node'].append(int(buffer))

                        elif state=='demand_data':
                            g = re.search('([0-9]*) ([0-9]*)',buffer)
                            data['demand'][g.group(1)] = g.group(2)

                        elif state=='edge_data':
                            tmp = buffer.split()
                            if specification['edge_data_format']==EdgeDataFormat.EDGE_LIST:
                                data['edge'][int(tmp[0])] = int(tmp[1])
                            elif specification['edge_data_format']==EdgeDataFormat.ADJ_LIST:
                                if tmp[-1]=='-1':
                                    data['edge'][int(tmp[0])] = [int(_x) for _x in tmp[1:-1]]
                                else:
                                    raise BadTspFile('Adjacent edge data is improperly terminated (no -1)')
                            else:
                                raise ParsingError('The parser got confused - unknown edge data format')

                        elif state=='fixed_edge_data':
                            tmp = buffer.split()
                            data['fixed_edges'][int(tmp[0])] = int(tmp[1])

                        elif state=='display_data':
                            if specification['display_data_type']==DisplayDataType.TWOD_DISPLAY:
                                tmp = buffer.split()
                                data['display'][int(tmp[0])] = [float(tmp[1]), float(tmp[2])]
                            else:
                                raise BadTspFile('Display data is provided when the display data type is not 2D')

                        elif state=='tour_data':

                            data['tour'] += [int(_x) for _x in buffer.split()]
                            if data['tour'][-1] == -1:
                                state = 'specification'
                                data['tour'].pop()


                        elif state=='edge_weight_data':
                            data['edge_weights'] += [float(_x) for _x in buffer.split()]
                        else:
                            raise ParsingError('The parser got itself into an unknown state')

    if 'dimension' not in specification:
        raise BadTspFile('Require dimension information')
    if 'node_coord_type' not in specification:
        specification['node_coord_type']=NodeCoordinateType.NO_COORDINATES

    # Return the parsed data to the caller.
    return specification, data



def load(filename: str, **kwargs) -> Union[Solution,Problem]:
    """Load and parse a TSPLIB file and call the appropriate generator to turn the dictionaries into objects.
    
    Additional options are passed directly to the parser (e.g., force_gzip).

    Parameters
    ----------
    filename : str
        Filename to load.

    Returns
    -------
    Union[Solution,Problem]
        Solution or Problem object.  None if there was some problem.

    Raises
    ------
    BadTspFile
        If the file is badly formatted, or doesn't contain the right information for the type.
    """
    result = None

    # Parse the TSPLIB file into two dictionaries.
    specification, data = parse_to_dict(filename, **kwargs)

    # Do some basic checks: we need a type.
    if 'type' not in specification:
        raise BadTspFile('Loaded file does not declare a type')

    # Check for a tour file - if it is then we construct a solution.
    if specification['type']=='TOUR':
        if 'tour' not in data:
            raise BadTspFile('Loaded a TOUR file without any tour data')

        result = generate_solution(specification, data)


    # Check for a symmetrical TSP - if it is we construct a problem.
    elif specification['type']==ProblemType.TSP:
        if ('edge_weight_type' not in specification):
            raise BadTspFile('Loaded a symmetrical TSP file without any edge weight specification')

        if (specification['edge_weight_type']==EdgeWeightType.EXPLICIT):
            if ('edge_weight_format' not in specification) or ('edge_weights' not in data):
                raise BadTspFile('Loaded a symmetrical TSP file with explicit edge weights but without a format or any data')
            result = generate_tsp(specification, data)
        else:
            if 'node_coord_type' not in specification:
                raise BadTspFile('Loaded a symmetrical TSP file with functional edge weights without any node coordinate type specification')
            if 'node_coords' not in data:
                raise BadTspFile('Loaded a symmetrical TSP file with functional edge weights without any node coordinate data')
            result = generate_tsp(specification, data)


    return result



def generate_tsp(specification: dict, data: dict) -> Problem:
    """Generate a Problem object with a TSP type from parsed dictionaries

    Parameters
    ----------
    specification : dict
        Dictionary of specification data.
    data : dict
        Dictionary of data detailing the problem.

    Returns
    -------
    Problem
        Object containing the problem data.

    Raises
    ------
    BadTspFile
        If there was a problem in the data.
    """

    result = Problem(ProblemType.TSP)

    result._edge_weight_type = specification['edge_weight_type']
    result._node_coord_type = specification['node_coord_type']
    result._name = specification['name']
    result._dimension = specification['dimension']

    if specification['node_coord_type']==NodeCoordinateType.TWOD:
        result._nodes = np.zeros((result._dimension,2), dtype=np.float32)
        if len(data['node_coords'])!=result._dimension:
            raise BadTspFile('Dimension does not match node coordinate length')
    elif specification['node_coord_type']==NodeCoordinateType.THREED:
        result._nodes = np.zeros((result._dimension,3), dtype=np.float32)
        if len(data['node_coords'])!=result._dimension:
            raise BadTspFile('Dimension does not match node coordinate length')
    else:
        pass

    if result._edge_weight_type==EdgeWeightType.ATT:
        result._metric = AttPseudoEuclidean()
    elif result._edge_weight_type==EdgeWeightType.CEILING_2D:
        result._metric = CeilEuclidean()
    elif result._edge_weight_type==EdgeWeightType.EUCLIDEAN_2D:
        result._metric = Euclidean()
    elif result._edge_weight_type==EdgeWeightType.EUCLIDEAN_3D:
        result._metric = Euclidean()
    elif result._edge_weight_type==EdgeWeightType.GEOGRAPHICAL:
        result._metric = Geographical()
    elif result._edge_weight_type==EdgeWeightType.MANHATTAN_2D:
        result._metric = Manhattan()
    elif result._edge_weight_type==EdgeWeightType.MANHATTAN_3D:
        result._metric = Manhattan()
    elif result._edge_weight_type==EdgeWeightType.MAXIMUM_2D:
        result._metric = MaximumLinf()
    elif result._edge_weight_type==EdgeWeightType.MAXIMUM_3D:
        result._metric = MaximumLinf()

    if 'node_coords' in data:
        for k,v in data['node_coords'].items():
            result._nodes[k-1] = v

    if 'edge_weights' in data:
        result._edge_weights = np.zeros((result._dimension,result._dimension))
        if specification['edge_weight_format']==EdgeWeightFormat.FULL_MATRIX:
            result._edge_weights = np.reshape(data['edge_weights'], (result._dimension,result._dimension))

        elif specification['edge_weight_format']==EdgeWeightFormat.UPPER_ROW:
            # No diagonal.
            w = result._dimension - 1
            j = 0
            for i in range(result._dimension-1):
                result._edge_weights[i,i+1:] = data['edge_weights'][j:(j+w)]
                j += w
                w -= 1
            result._edge_weights = result._edge_weights + result._edge_weights.T

        elif specification['edge_weight_format']==EdgeWeightFormat.LOWER_ROW:
            raise UnsupportedEdgeWeight()

        elif specification['edge_weight_format']==EdgeWeightFormat.UPPER_DIAG_ROW:
            raise UnsupportedEdgeWeight()

        elif specification['edge_weight_format']==EdgeWeightFormat.LOWER_DIAG_ROW:
            # With diagonal.
            w = 1
            j = 0
            for i in range(result._dimension):
                result._edge_weights[i,:w] = data['edge_weights'][j:(j+w)]
                j += w
                w += 1
            result._edge_weights = result._edge_weights + result._edge_weights.T

        elif specification['edge_weight_format']==EdgeWeightFormat.UPPER_COL:
            raise UnsupportedEdgeWeight()

        elif specification['edge_weight_format']==EdgeWeightFormat.LOWER_COL:
            raise UnsupportedEdgeWeight()

        elif specification['edge_weight_format']==EdgeWeightFormat.UPPER_DIAG_COL:
            raise UnsupportedEdgeWeight()

        elif specification['edge_weight_format']==EdgeWeightFormat.LOWER_DIAG_COL:
            raise UnsupportedEdgeWeight()

    return result


def generate_solution(specification: dict, data: dict) -> Solution:
    """Generate a solution object containing tour data parsed from dictionaries

    Parameters
    ----------
    specification : dict
        Dictionary of specification data.
    data : dict
        Dictionary of data detailing the tour.


    Returns
    -------
    Solution
        Object containing the tour data.

    Raises
    ------
    BadTspFile
        If there was a problem in the data.
    """

    result = Solution()

    if 'dimension' in specification:
        if specification['dimension']!=len(data['tour']):
            raise BadTspFile('Length of tour data does not match dimension')
        result.dimension = specification['dimension']+1
    else:
        result.dimension = len(data['tour'])+1

    if 'name' in specification:
        result.name = specification['name']

    [result.append_comment(_x) for _x in specification['comments']]

    result.data = np.array(data['tour'])-1

    return result
