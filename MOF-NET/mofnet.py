import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
 
class SelfWeight(nn.Module):
    """
    Simple feedforward network that learns weights using topological embeddings 
    to update the node (SBU) and edge (linkers) representation.
    """

    def __init__(self,topo_emb_size, node_emb_size, out_size):
        
        super().__init__()
    
        self.node_emb_size = node_emb_size 
        self.out_size = out_size

        input_size = topo_emb_size  
        output_size = 3 * node_emb_size * out_size 
        
        self.dense = nn.Linear(input_size,output_size) 
        self.act = nn.Tanh()

    def forward(self, x):

        B = x.shape[0]
     
        x = self.dense(x)
        x = self.act(x)

        weight = torch.reshape(x, (B, 3, self.out_size, self.node_emb_size))  

        return weight


class InteractionWeight(nn.Module):
    """
    Simple feedforward network that learns weights using topological embeddings 
    to update the concatinated MOF representation.
    """
    
    def __init__(self, topo_emb_size,in_size, out_size):
    
        super().__init__()

        self.in_size = in_size 
        self.out_size = out_size 

        input_size = topo_emb_size  
        output_size = in_size * out_size 

        self.dense = nn.Linear(input_size,output_size)
        self.activation = nn.Tanh()

    def forward(self, x):

        B = x.shape[0]

        x = self.dense(x)
        x = self.activation(x)
       
        weight = torch.reshape(x,(B, self.in_size, self.out_size)) 

        return weight


class PredictionNN(nn.Module):
    """
    Simple feedforward prediction network with input MOF graph representation and working capacity target.
    """

    def __init__(self,mof_emb_size):
        """ Arhitecture of prediction NN 
        
        Args:
           -mof_emb_size (int): size of input graph represenation vector
        """

        super().__init__()

        input_size = mof_emb_size
        output_size = 32
        
        self.dense = nn.Linear(input_size,output_size,bias=False) 
        self.hidden_batchnorm = nn.BatchNorm1d(output_size) 
        self.activation = nn.ReLU()
        self.output_dense = nn.Linear(output_size,1) 


    def forward(self, x): 
        """ Forward pass through the GNN.
        
        Args:
            -x (torch.Tensor): graph represenation vector of input MOF

        Returns:
            -working_cap (torch.Tensor): tensor containing working capacity prediction for input MOF
        """

        x = self.dense(x) 
        x = self.hidden_batchnorm(x) 
        x = self.activation(x)
        working_cap = self.output_dense(x) 

        return working_cap



class mofnet(nn.Module):
    """
    The full MOF-NET model.
    """

    def __init__(self, **kwargs):
        """ Arhitecture of MOF-NET.

        Args
            -kwargs (optional): keyword arguments to override default hyperparameters
        """

        super().__init__()

        # Default hyperparameters
        default_parameters = {
            "num_topos": 3000,
            "num_nodes": 1000,
            "num_edges": 300,
            "topo_emb_size": 128,
            "node_emb_size": 128,
            "edge_emb_size": 128,
            "node_self_size": 128,
            "edge_self_size": 128,
            "mof_emb_size": 64,
            "dropout_rate": 0.5,
            "activation": F.relu
        }

        # Override defaults with any provided keyword arguments
        params = {}
        for key in default_parameters.keys():
            if key in kwargs:
                params[key] = kwargs[key]
            else:
                params[key] = default_parameters[key]

        # Unpack parameters
        num_topos = params["num_topos"]
        num_nodes = params["num_nodes"]
        num_edges = params["num_edges"]
        topo_emb_size = params["topo_emb_size"]
        node_emb_size = params["node_emb_size"]
        edge_emb_size = params["edge_emb_size"]
        node_self_size = params["node_self_size"]
        edge_self_size = params["edge_self_size"]
        mof_emb_size = params["mof_emb_size"]
        dropout_rate = params["dropout_rate"]
        self.activation = params["activation"]

        # instantiate NN for self-learnt represenation (node,edges and toplogy)              
        self.topo_embedding = nn.Embedding(num_embeddings=num_topos, embedding_dim=topo_emb_size)
        self.node_embedding = nn.Embedding(num_embeddings=num_nodes, embedding_dim=node_emb_size)
        self.edge_embedding = nn.Embedding(num_embeddings=num_edges, embedding_dim=edge_emb_size)

        # instantiate NN for toplogy weight incorporation                
        self.node_weight = SelfWeight(topo_emb_size,node_emb_size, node_self_size)
        self.edge_weight = SelfWeight(topo_emb_size,edge_emb_size, edge_self_size)
        first_size = 3*node_self_size + 3*edge_self_size
        self.interaction_weight = InteractionWeight(topo_emb_size,first_size, mof_emb_size)
        
        self.dropout = nn.Dropout(p=dropout_rate)

        self.node_batchnorm = nn.BatchNorm1d(3*node_self_size)
        self.edge_batchnorm = nn.BatchNorm1d(3*edge_self_size) 
        self.interaction_batchrnom = nn.BatchNorm1d(mof_emb_size)

        # instantiate prediction NN
        self.prediction_nn = PredictionNN(mof_emb_size)


    def calculate_mof_embedding(self, x): 
        """ Calculation of MOF representation.
            
        Args:
            -x (torch.Tensor): MOF input made up of an ID number for toplogy, 3 nodes (SBUs) and  3 linkers (linkers).

        Returns:
            -x (torch.Tensor): graph represenation vector of input MOF
        """

        B = x.shape[0] # B: batch size.


        # Split input vector to topology, nodes and edges.
        topo_x, node_x, edge_x = torch.split(x, [1, 3, 3], dim=1)

        
        # Make topology embedding  
        topo_emb = self.topo_embedding(topo_x)
        topo_emb = torch.reshape(topo_emb, [B, -1])
        topo_emb = self.dropout(topo_emb)


        # Make node embedding
        node_x = torch.where(node_x >= 0, node_x, torch.zeros_like(node_x))
        node_emb = self.node_embedding(node_x)
        node_emb = self.dropout(node_emb)
        # Apply self interaction in topology.
        # node_weight: (B, 3, Eout, E), node_emb: (B, 3, E).
        # Result: (B, 3, Eout).
        node_weight = self.node_weight(topo_emb)
        node_emb = torch.einsum("ijkl,ijl->ijk", node_weight, node_emb)
        node_emb = self.node_batchnorm(node_emb.reshape([B,-1]))
        node_emb = self.activation(node_emb).reshape([B,3,self.node_weight.out_size]) 
        # Apply mask and reshape to [B, 3*node_self_size]
        shape = list(node_x.shape) + [self.node_weight.out_size]
        mask = node_x.unsqueeze(-1).expand(*shape)  
        mask = mask >= 0
        node_emb = torch.where(mask, node_emb, torch.zeros_like(node_emb))
        node_emb = node_emb.view(B, -1)

        
        # Make edge embedding
        edge_x = torch.where(edge_x >= 0, edge_x, torch.zeros_like(edge_x))
        edge_emb = self.edge_embedding(edge_x)
        edge_emb = self.dropout(edge_emb)
        # Apply self interaction in topology.
        # edge_weight: (B, 3, Eout, E), edge_emb: (B, 3, E).
        # Result: (B, 3, Eout).
        edge_weight = self.edge_weight(topo_emb)
        edge_emb = torch.einsum("ijkl,ijl->ijk", edge_weight, edge_emb)
        edge_emb = self.edge_batchnorm(edge_emb.reshape([B,-1]))
        edge_emb = self.activation(edge_emb).reshape([B,3,self.edge_weight.out_size]) 
        # Apply mask and reshape to [B, 3*node_self_size]
        shape = list(edge_x.shape) + [self.edge_weight.out_size]  
        mask = edge_x.unsqueeze(-1).expand(*shape)                
        mask = mask >= 0
        edge_emb = torch.where(mask, edge_emb, torch.zeros_like(edge_emb))  
        edge_emb = edge_emb.view(B, -1) 


        # Concatenate building block tensors.
        # x: [B, 3 * Nout + 3 * Eout]
        x = torch.cat([node_emb, edge_emb], dim=1)
      

        # Apply interaction.
        weight = self.interaction_weight(topo_emb) # x: [B, 3 * Nout + 3 * Eout]
        x = torch.einsum("ijk,ij->ik", weight, x)  # [B, output_dim]
        x = self.interaction_batchrnom(x) 
        x = self.activation(x)
     
        return x


    def forward(self, x): 
        """ Forward pass through MOF-NET.
        
        Args:
            -x (torch.Tensor): MOF input made up of an ID number for toplogy, 3 nodes (SBUs) and  3 linkers (linkers).

        Returns:            
            -working_cap (torch.Tensor): tensor containing working capacity prediction for input MOF

        """


        x = self.calculate_mof_embedding(x) 
        working_cap = self.prediction_nn(x)

        return working_cap



