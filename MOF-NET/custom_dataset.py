import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

import os.path


class custom_dataset(Dataset):

    def __init__(self,file_path):
        """ Custom PyTorch Dataset.
        
        Args:  
            -file (str): string containing file path to read data from.
        """

        # Read file
        file = pd.read_csv(file_path,delimiter=" ",header=None)

        # Remove data points with missing xyz files (for comparision with MOF-NET-G which requires xyz files)
        data = np.load("./data_loader_state-20200116.npz", allow_pickle=True) 
        node_hash = data["node_hash"].tolist()
        edge_hash = data["edge_hash"].tolist()
        missing = []
        for node_id in node_hash.keys():
            if not os.path.isfile("../MOF-NET-G/data/bbs/" + node_id + ".xyz"):
                missing.append(node_id)
        for edge_id in edge_hash.keys():
            if not os.path.isfile("../MOF-NET-G/data/bbs/" + edge_id + ".xyz"):
                missing.append(edge_id)
        for m in missing:
            file = file[~file[0].str.contains(m)]
       

        # Extract MOFs and working capacity label from file 
        mof = file.iloc[:,0] 
        working_cap = file.iloc[:,14] - file.iloc[:,3] 
        working_cap = working_cap/100.0 

        mof_inputs = []
        for i in range(len(mof)):
            mof_inputs.append(self.key2index(mof.iloc[i]))

        self.x = torch.tensor(mof_inputs)  
        self.y = torch.tensor(working_cap.values) 
      
      
    def key2index(self,key):
        """ Converts MOF key string into indices from precomputed hash dictionaries.

        Args:
            key (str): string representing the input MOF. Contains ID of toplogy,nodes(SBUs) and edges(linkers) used e.g "ZEO+N1+N2+N3+E1+E2+E3"

        Returns:
            [int,list[int],list[int]]: list of indices
        """

        #fill hash dicts 
        data = np.load("data_loader_state-20200116.npz", allow_pickle=True) 
        topo_hash = data["topo_hash"].item()
        node_hash = data["node_hash"].tolist()
        edge_hash = data["edge_hash"].tolist()

        tokens = key.split("+")

        nodes = [v for v in tokens if v.startswith("N")]
        edges = [v for v in tokens if v.startswith("E")]

        topo_index = [topo_hash[tokens[0]]]

        node_index = [-1] * 3
        for i, v in enumerate(nodes):
            node_index[i] = node_hash[v]

        edge_index = [-1] * 3
        for i, v in enumerate(edges):
            edge_index[i] = edge_hash[v]
        
        index = topo_index + node_index + edge_index

        return index


    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return self.x[idx].int(),self.y[idx].float()




