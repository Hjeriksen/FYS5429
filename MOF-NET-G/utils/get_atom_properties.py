import pandas as pd

def lookup_table(atom,attributes):
	""" A function that takes an atom symbol and returns a list of its properties from a CSV file.
    
    Args:
       	-atom (str): A string representing the atomic symbol 
    	-attributes (list[str]): A list of column names (attribute names) to extract for the atom.
    
    Returns:
        -extracted_attributes (list[float]): List of property values for the given atom and attributes.
    """
	
	# Read atom properties CSV
	file = pd.read_csv("./utils/atom_properties.csv") 
	
	# Extract value for the attributes 
	atom_row = file[file["'Symbol'"]==atom]

	extracted_attributes = []
	for i in attributes:
		attribute=float(atom_row[i].item())
		extracted_attributes.append(attribute)
		
		
	return extracted_attributes
