import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv,Set2Set

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
    Simple feedforward prediction network with input MOF graph representation a
    nd working capacity target.
    """

    def __init__(self,mof_emb_size):
        """Arhitecture of prediction NN.
        
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
        """ Forward pass through the GNN
        
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


class GNN(nn.Module):
    """
    A graph neural network using NNConv layers, GRU for message passing,
    and Set2Set for graph-level readout.
    """
    
    def __init__(self,n_node_features,n_edge_features,dim,set2set_steps=3,depth=3):
        """ Arhitecture of GNN.
        
        Args:
            -n_node_features (int): dimension of node feature vectors
            -n_edge_features (int): dimension of edge feature vectors
            -dim (int): size of produced graph feature vector
            -depth (int): Message passing depth
        """

        super().__init__()
        
        self.depth = depth 
        dim = int(dim/2)

        #embed node features 
        self.node_feat_embedding = nn.Linear(n_node_features,dim)
        #embed edge features 
        edge_feat_embedding = nn.Sequential(nn.Linear(n_edge_features,128),nn.ReLU(),nn.Linear(128,dim*dim))
        
        #convolution layer for message passing
        self.conv = NNConv(dim,dim,edge_feat_embedding, aggr="mean")
        #update node features across multiple message passing iterations
        self.gru = nn.GRU(dim,dim)

        self.set2set = Set2Set(dim,processing_steps=set2set_steps)

    def forward(self, data):
        """ Forward pass through the GNN.
        
        Args:
            -data (torch_geometric.data.Data): Input pytorch graph. 

        Returns:
            -graph_rep_vector (torch.Tensor): Graph-level representation. 
        """

        #embed node feature vectors
        node_feature_vectors = F.relu(self.node_feat_embedding(data.x))
        h = node_feature_vectors.unsqueeze(0)

        #perform multiple message-passing iterations
        for _ in range(self.depth):
            #embed edge feature vectors and compute messages 
            messages = F.relu(self.conv(node_feature_vectors,data.edge_index,data.edge_attr))
            
            #update node features using GRU
            node_feature_vectors,h = self.gru(messages.unsqueeze(0),h)
            node_feature_vectors = node_feature_vectors.squeeze(0)
        
        #readout: aggregate node features into graph-level feature vector using Set2Set
        graph_rep_vector = self.set2set(node_feature_vectors.to(device), data.batch.to(device))

        return graph_rep_vector
    

def dynamic_offset_reorder(tensor: torch.Tensor, offset: int) -> torch.Tensor:
    """ Reorder rows of a 2D tensor with a dynamic stride-based offset.
    
    Args:
        -tensor (torch.Tensor): Input tensor of shape (N, D)
        -offset (int): The dynamic stride or offset

    Returns:
        -torch.Tensor: Reordered tensor
    """
    n_rows = tensor.size(0)
    reordered_indices = []

    # Build index groups by offset
    for start in range(offset):
        indices = torch.arange(start, n_rows, offset)
        reordered_indices.append(indices)

    # Concatenate all index groups
    final_order = torch.cat(reordered_indices)
    return tensor[final_order]



class mofnet_GNN(nn.Module):
    """
    The full MOF-NET-G model.
    """

    def __init__(self, **kwargs):        
        """ Arhitecture of MOF-NET-G.

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

        # instantiate NN for self-learnt toplogy represenation                
        self.topo_embedding = nn.Embedding(num_embeddings=num_topos, embedding_dim=topo_emb_size)
    
        # instantiate NN for toplogy weight incorporation                
        self.node_weight = SelfWeight(topo_emb_size,node_emb_size, node_self_size)
        self.edge_weight = SelfWeight(topo_emb_size,edge_emb_size, edge_self_size)
        first_size = 3*node_self_size + 3*edge_self_size

        self.interaction_weight = InteractionWeight(topo_emb_size,first_size, mof_emb_size)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.node_batchnorm = nn.BatchNorm1d(3*node_self_size) 
        self.edge_batchnorm = nn.BatchNorm1d(3*edge_self_size)
        self.interaction_batchrnom = nn.BatchNorm1d(mof_emb_size)

        # Instantiate GNN
        self.node_emb_size = node_emb_size
        self.edge_emb_size = edge_emb_size

        self.NGNN = GNN(n_node_features=2,n_edge_features=5,dim=node_emb_size)
        self.EGNN = GNN(n_node_features=2,n_edge_features=5,dim=node_emb_size)

        # instantiate prediction NN
        self.prediction_nn = PredictionNN(mof_emb_size)


    def calculate_mof_embedding(self, x): 
        """ Calculation of MOF representation.
            
        Args:
            -x (list[int,list[pytorch_geometric.data.Data],list[pytorch_geometric.data.Data]]):MOF input made up 
            of an ID number for toplogy and lists containing 3 graphs for the nodes (SBUs) and linkers (linkers)

        Returns:
            -x (torch.Tensor): graph represenation vector of input MOF
        """

        # Unpack datapoint
        topo = x[0]
        node_graphs = x[1]
        edge_graphs = x[2]

        B = topo.shape[0]  # B: batch size.

        # Make topology embedding 
        topo_emb = self.topo_embedding(topo)
        topo_emb = torch.reshape(topo_emb, [B, -1])
        topo_emb = self.dropout(topo_emb)

        # Make node embedding
        node_emb = self.NGNN(node_graphs) #create graph representation vectors for the nodes (SBUs)
        node_emb = dynamic_offset_reorder(node_emb, offset=B) #reorganize batch
        node_emb = node_emb.reshape(B,3,self.node_emb_size)
        node_emb = self.dropout(node_emb)
        # Apply self interaction in topology.
        # node_weight: (B, 3, Eout, E), node_emb: (B, 3, E).
        # Result: (B, 3, Eout).
        node_weight = self.node_weight(topo_emb)
        node_emb = torch.einsum("ijkl,ijl->ijk", node_weight, node_emb)
        node_emb = self.node_batchnorm(node_emb.reshape([B,-1]))
        node_emb = self.activation(node_emb).reshape([B,3,self.node_weight.out_size]) 
        #reshape to [B, 3*node_self_size]   
        node_emb = node_emb.view(B, -1)
   
        # Make edge embedding 
        edge_emb = self.EGNN(edge_graphs) #create graph representation vectors for the edges (linkers)
        edge_emb = dynamic_offset_reorder(edge_emb, offset=B) #reorganize batch
        edge_emb = edge_emb.reshape(B,3,self.edge_emb_size) 
        edge_emb = self.dropout(edge_emb)
        # Apply self interaction in topology.
        # edge_weight: (B, 3, Eout, E), edge_emb: (B, 3, E).
        # Result: (B, 3, Eout).
        edge_weight = self.edge_weight(topo_emb)
        edge_emb = torch.einsum("ijkl,ijl->ijk", edge_weight, edge_emb)
        edge_emb = self.edge_batchnorm(edge_emb.reshape([B,-1]))
        edge_emb = self.activation(edge_emb).reshape([B,3,self.edge_weight.out_size])
        #reshape to [B, 3*node_self_size]        
        edge_emb = edge_emb.view(B, -1) 


        # Concatenate building block tensors
        # x: [B, 3 * Nout + 3 * Eout]
        x = torch.cat([node_emb, edge_emb], dim=1)
       
        # Apply interaction.
        weight = self.interaction_weight(topo_emb) # x: [B, 3 * Nout + 3 * Eout]
        x = torch.einsum("ijk,ij->ik", weight, x)  # [B, output_dim]
        x = self.interaction_batchrnom(x) 
        x = self.activation(x)
     
        return x


    def forward(self, x): 
        """ Forward pass through MOF-NET-G.
        
        Args:
            -x (list[int,list[pytorch_geometric.data.Data],list[pytorch_geometric.data.Data]]):MOF input made up 
            of an ID number for toplogy and lists containing 3 graphs for the nodes (SBUs) and linkers (linkers)

        Returns:            
            -working_cap (torch.Tensor): tensor containing working capacity prediction for input MOF

        """

        x = self.calculate_mof_embedding(x) 
        working_cap = self.prediction_nn(x)

        return working_cap

