import os.path
import numpy as np

from utils.xyz_to_graph import graph_from_file

def lookup_graph(fragment_type):
    """ Create lookup tables for all buidling block graphs, i.e node (SBU) and edge (linker) buidling blocks respectivly, 
    to only read once. Each uniqe fragment (node or edge) is associated with a unique ID and reused across MOF datapoints.

    Args:
        -fragment_type (str): "N" for nodes, "E" for edges.

    Returns:
        -graphs (dict): Dictionary mapping fragment ID to its graph object.
        -missing (list[str]]): List of fragment names (e.g., "N23") that had missing .xyz files.
    """

    # Load saved hash mappings
    data = np.load("./data/data_loader_state-20200116.npz", allow_pickle=True) 
   
    graphs = {}
    missing = []
    
    # Handle node (SBU) fragments
    if fragment_type == "N":

        node_hash = data["node_hash"].tolist()

        for node_id in node_hash.values():
            filepath = "./data/bbs/" + "N" + str(node_id) + ".xyz"
            if os.path.isfile(filepath):
                graph = graph_from_file(filepath)
                graphs[node_id] = graph
            else:
                missing.append("N" + str(node_id))
    # Handle edge (linker) fragments  
    else:

        edge_hash = data["edge_hash"].tolist()

        for edge_id in edge_hash.values():
            filepath = "./data/bbs/" + "E" + str(edge_id) + ".xyz"
            if os.path.isfile(filepath):
                graph = graph_from_file(filepath)
                graphs[edge_id] = graph
            else:
                missing.append("E" + str(edge_id) )

    return graphs,missing