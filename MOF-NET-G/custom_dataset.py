import copy

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

import pandas as pd
import numpy as np


class custom_graph_dataset(Dataset):
  
    def __init__(self, file,graphs_node_dict,graphs_edge_dict,scaling=[]):
        """ Custom PyTorch Dataset to handle MOF graph data.
        
        Args:  
            -file (pandas.DataFrame): DataFrame containing MOF keys and target values.            
            -graphs_node_dict (dict): Dictionary mapping node graph keys to preloaded PyTorch Geometric graphs.
            -graphs_edge_dict (dict): Dictionary mapping edge graph keys to preloaded PyTorch Geometric graphs.
            -scaling (list[torch.Tensor]): list of mean and std tensors from training data points to scale validation and test data points.
        """

        # Extract MOFs from file and use lookup table to extract corresponding graphs 
        mof = file.iloc[:,0] 
        
        topo_indices = []
        node_indices = []
        edge_indices = []
        for i in range(len(mof)):
            topo_index, node_index, edge_index = self.key2index(mof.iloc[i]) 

            topo_indices.append(topo_index)
            node_indices.append(node_index)
            edge_indices.append(edge_index)

        self.node_graphs = [[copy.deepcopy(graphs_node_dict[i]) if i != -1 else self._dummy_graph() for i in node_ids] for node_ids in node_indices] #add deepcopy to avoid pass-by-refrence
        self.edge_graphs = [[copy.deepcopy(graphs_edge_dict[i]) if i != -1 else self._dummy_graph() for i in edge_ids] for edge_ids in edge_indices]
    
    
        # Scale the node and edge feature vectors in the graphs (of SBUs and linkers).

        if len(scaling) == 0: #for custom training dataset: calulate training metrics (mean and STD) used to scale train,valdidate and test
            #for the nodes (SBUs)
            graph_node_feature_list_N = []
            graph_edge_feature_list_N = []
            for mof_node_tuple in self.node_graphs:
                for node in mof_node_tuple:
                    graph_node_feature_list_N.extend(node.x.detach().numpy().tolist())
                    graph_edge_feature_list_N.extend(node.edge_attr.detach().numpy().tolist())
            self.graph_node_feature_list_N_mean = torch.mean(torch.tensor(graph_node_feature_list_N),0)
            self.graph_node_feature_list_N_std = torch.std(torch.tensor(graph_node_feature_list_N),0)
            self.graph_edge_feature_list_N_mean = torch.mean(torch.tensor(graph_edge_feature_list_N),0)
            self.graph_edge_feature_list_N_std = torch.std(torch.tensor(graph_edge_feature_list_N),0)

            #for the edges (linkers)
            graph_node_feature_list_E =  []
            graph_edge_feature_list_E = []
            for mof_edge_tuple in self.edge_graphs:
                for edge in mof_edge_tuple:
                    graph_node_feature_list_E.extend(edge.x.detach().numpy().tolist())
                    graph_edge_feature_list_E.extend(edge.edge_attr.detach().numpy().tolist())
            self.graph_node_feature_list_E_mean = torch.mean(torch.tensor(graph_node_feature_list_E),0)
            self.graph_node_feature_list_E_std = torch.std(torch.tensor(graph_node_feature_list_E),0)
            self.graph_edge_feature_list_E_mean = torch.mean(torch.tensor(graph_edge_feature_list_E),0)
            self.graph_edge_feature_list_E_std = torch.std(torch.tensor(graph_edge_feature_list_E),0)
        else: #for custom validation and test dataset: load in training metrics (mean and STD) 
            self.graph_node_feature_list_N_mean = scaling[0]
            self.graph_node_feature_list_N_std = scaling[1]
            self.graph_edge_feature_list_N_mean = scaling[2]
            self.graph_edge_feature_list_N_std = scaling[3]
            
            self.graph_node_feature_list_E_mean = scaling[4]
            self.graph_node_feature_list_E_std = scaling[5]
            self.graph_edge_feature_list_E_mean = scaling[6]
            self.graph_edge_feature_list_E_std = scaling[7]
      
        #apply scaling of feature vectors of SBU and linker graphs respectivly
        for i, _ in enumerate(self.node_graphs):
                for j, __ in enumerate(self.node_graphs[i]):
                    
                    self.node_graphs[i][j].x = (self.node_graphs[i][j].x - self.graph_node_feature_list_N_mean) / self.graph_node_feature_list_N_std
                    self.node_graphs[i][j].edge_attr = (self.node_graphs[i][j].edge_attr - self.graph_edge_feature_list_N_mean) / self.graph_edge_feature_list_N_std
       
        for i, _ in enumerate(self.edge_graphs):
                for j, __ in enumerate(self.edge_graphs[i]):
                    
                    self.edge_graphs[i][j].x = (self.edge_graphs[i][j].x - self.graph_node_feature_list_E_mean) / self.graph_node_feature_list_E_std
                    self.edge_graphs[i][j].edge_attr = (self.edge_graphs[i][j].edge_attr - self.graph_edge_feature_list_E_mean) / self.graph_edge_feature_list_E_std


        # Extract working capacitiy label from file
        working_cap = file.iloc[:,14] - file.iloc[:,3] 
        working_cap = working_cap/100.0 


        self.topo = torch.tensor(topo_indices)
        self.y = torch.tensor(working_cap.values) 
    

    def _dummy_graph(self):
        """
        Creates a dummy/empty pytorch graph with empty edge list and 0-valued node/edge features.
        Used when a buidling block (framework calls for 3 SBUs and 3 linkers) is missing.
        """

        return Data(
            x=torch.zeros(1, 2),  
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, 5)  
        )
    
   
    def get_train_metrics(self):
        """
        Returns scaling statistics to be reused for validation/test sets.
        """
        
        return [
            self.graph_node_feature_list_N_mean,
            self.graph_node_feature_list_N_std,
            self.graph_edge_feature_list_N_mean,
            self.graph_edge_feature_list_N_std,
            self.graph_node_feature_list_E_mean,
            self.graph_node_feature_list_E_std,
            self.graph_edge_feature_list_E_mean,
            self.graph_edge_feature_list_E_std,
        ]
   

    def key2index(self,key):
        """ Converts MOF key string into indices from precomputed hash dictionaries.

        Args:
            key (str): string representing the input MOF. Contains ID of toplogy,nodes(SBUs) and edges(linkers) used e.g "ZEO+N1+N2+N3+E1+E2+E3"

        Returns:
            [int,list[int],list[int]]: list of indices
        """

        #fill hash dicts 
        data = np.load("./data/data_loader_state-20200116.npz", allow_pickle=True) 
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
        
        return topo_index, node_index, edge_index


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.topo[idx].int(), self.node_graphs[idx], self.edge_graphs[idx], self.y[idx].float()




