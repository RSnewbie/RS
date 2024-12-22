import torch
import math
import numpy as np
from torch import nn
import scipy.sparse as sp
import networkx as nx
import torch.nn.functional as F
from torch_scatter import scatter
from torch.nn.parameter import Parameter
from scipy.sparse import coo_matrix,hstack
from torch_geometric.utils import degree
from common.abstract_recommender import GeneralRecommender

bet_ = []

class New_idx(GeneralRecommender):
    def __init__(self, config, dataset):
        super(New_idx, self).__init__(config, dataset)

    def degree_drop_weights(self,relation):
        deg = degree(relation[1])
        deg_col = deg[relation[1]]
        s_col = deg_col
        weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())    
        return weights

    def pr_drop_weights(self,pr):
        s = pr
        return (s.max() - s) / (s.max() - s.float().mean())

    def evc_drop_weights(self,evc):
        s = evc
        return (s.max() - s) / (s.max() - s.float().mean())
    
    def drop_edge_weighted(self,edge_weights,relation, p: float, threshold: float = 1.):
        edge_weights = edge_weights / edge_weights.mean() * p
        edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
        sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)    
        return sel_mask
    
    def idx_sel(self,flag,weight,rela,keep):
        if flag == 'node_drop':
            drop_weights = self.degree_drop_weights(rela)
            drop_weights = drop_weights.squeeze()
            new_adj = self.drop_edge_weighted(drop_weights,rela, p=0.9, threshold=1)   
            return new_adj
        elif flag == 'edge_drop':
            drop_weights = self.pr_drop_weights(weight)
            new_adj = self.drop_edge_weighted(drop_weights,rela, p=0.1, threshold=0.9)    
            return new_adj
        elif flag == 'random_walk':
            self.drop_weights = self.evc_drop_weights(weight)
            self.drop_weights = self.drop_weights.squeeze()
            new_adj = self.drop_edge_weighted(self.drop_weights,rela, p=0.1, threshold=0.9)    
            return new_adj
