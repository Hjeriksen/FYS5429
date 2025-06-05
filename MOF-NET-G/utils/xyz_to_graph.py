import torch
from torch_geometric.data import Data

from utils.get_atom_properties import lookup_table


def calculate_euclidean_distance(v1,v2):

    """Calculates the euclidean distance between 2 atoms (from positions in xyz file given with 4 digits (float))
    
    Args:
    	-v1 (list[float]): list of position (x,y,z) of atom 1 (in Å)
    	-v2 (list[float]): list of position (x,y,z) of atom 2 (in Å)
    	
    Return:
    	-(float): distance between position (in Å)
    """
    
    summation = 0
    for i in range(len(v1)):
        summation += (v1[i] - v2[i]) ** 2

    return summation**0.5


def parse_xyz(file): 
    """Parses xyz file into a pytorch geometric graph.
    The xyz files contains the xyz positions of the atoms and an adjecency list with bond orders.

    Returns:
        -graph (torch_geometric.data.Data): a pytorch geometric graph representing the molecule
    """

    # Skip number of atoms and comment line of xyz file
    file.readline()
    file.readline()

    # Read atoms, adjeceny list and bond type from xyz file
    elements = []
    positions = []

    adj_list = []
    edge_attr= []
    one_hot_bond_idx = {"S":0,"D":1,"T":2,"A":3} # Single, Double, Triple, Aromatic

    for line in file:

        line =  line.strip()

        if line != "": 
            line_split = line.split()
            
            # Read atoms (element and position) - top of xyz file
            if len(line_split) == 4:

                if line_split[0]!= "X": #skip reading connecting atoms with symbol "X" 

                    elements.append(line_split[0])
                    positions.append([
                        float(line_split[1]),
                        float(line_split[2]),
                        float(line_split[3])
                        ])
                
            # Read adjeceny list and bond type - bottom of xyz file
            else:
                node1_idx = int(line_split[0])
                node2_idx = int(line_split[1])

                if node1_idx< len(elements) and node2_idx < len(elements): #skip reading connecting atoms with symbol "X" 

                    adj_list.append([node1_idx,node2_idx])
                    adj_list.append([node2_idx,node1_idx]) #pytorch requires directed graphs. Add to make undirected

                    # Determine edge features (bond type (one hot encoding) and euclidian distance) 
                    distance = calculate_euclidean_distance(positions[node1_idx], positions[node2_idx])
                    
                    bond_type = line_split[2]
                    bond_type_onehot = [0,0,0,0] 
                    bond_type_onehot[one_hot_bond_idx[bond_type]] = 1

                    edge_attr.append(bond_type_onehot+[distance])
                    edge_attr.append(bond_type_onehot+[distance]) #pytorch requires directed graphs. Add to make undirected
    
    # Determine node features (electronegativity and covalent radius) from csv file 
    node_attr = []
    attributes = ["'Electronegativity (Pauling)'","'Covalent radius’"]
        	
    for atom in elements:
        atom = "'" + atom + "'"
       
        extracted_attributes = lookup_table(atom, attributes)
        node_attr.append(extracted_attributes) 
    
    
    # Convert to tensors and create pytorch graph
    edge_index= torch.tensor(adj_list)
    edge_attr= torch.tensor(edge_attr)
    node_attr= torch.tensor(node_attr)
    
    graph = Data(x=node_attr, edge_index=edge_index.t().contiguous(), edge_attr = edge_attr) 

    return graph 
    

def graph_from_file(filename):
    """ Wrapper to open and parse a xyz file into a pytorch geometric graph. 
    
    Args:
        -filename (str): Path to the xyz file

    Returns:
        -graph (torch_geometric.data.Data): a pytorch geometric graph representing the molecule
    """

    with open(filename, 'r') as f:
       graph = parse_xyz(f) 
    
    return graph



