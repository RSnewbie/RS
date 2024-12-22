import math
import torch as t
import numpy as np
from torch import nn
from torch.nn import init
import dgl.function as fn
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from models.centrality import New_idx
from common.abstract_recommender import GeneralRecommender
from torch_sparse import SparseTensor
"""
Graph Related Augmentation
"""
class EdgeDrop(nn.Module):
    """ Drop edges in a graph.
    """
    def __init__(self, resize_val=False):
        super(EdgeDrop, self).__init__()
        self.resize_val = resize_val

    def forward(self, adj, keep_rate):
        """
        :param adj: torch_adj in data_handler
        :param keep_rate: ratio of preserved edges
        :return: adjacency matrix after dropping edges
        """
        if keep_rate == 1.0: return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = (t.rand(edgeNum) + keep_rate).floor().type(t.bool)
        newVals = vals[mask] / (keep_rate if self.resize_val else 1.0)
        newIdxs = idxs[:, mask]
        return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)

class New_EdgeDrop(GeneralRecommender):
    """ Drop edges in a graph.
    """
    def __init__(self, config, dataset):
        super(New_EdgeDrop, self).__init__(config, dataset)
        self.user_num = self.n_users
        self.item_num = self.n_items
        self.idx = New_idx(config, dataset)

    def trn_coo(self,matrix,user_num,item_num):
        rows = matrix[0].cpu().numpy()
        cols = matrix[1].cpu().numpy()
        data = np.ones(rows.size)
        coo_matrix = sp.coo_matrix((data, (rows, cols)), shape=(user_num, item_num))
        return coo_matrix

    def _normalize_adj(self, mat):

        # Add epsilon to avoid divide by zero
        degree = np.array(mat.sum(axis=-1)) + 1e-10
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()

    def _make_torch_adj(self, mat):

        a = csr_matrix((self.user_num, self.user_num))
        b = csr_matrix((self.item_num, self.item_num))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        # mat = (mat + sp.eye(mat.shape[0])) * 1.0# MARK
        mat = self._normalize_adj(mat)

        # make torch tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).to(self.device)

    def forward(self, adj,temp, keep_rate, augmentation):
        """
        :param embeds: the embedding matrix of nodes in the graph
        :param keep_rate: ratio of preserved nodes
        :return: the embeddings matrix after dropping nodes
        """
        self.augmentation = augmentation
        self.result = self.idx.idx_sel(self.augmentation,temp,adj,keep_rate)
        self.before_adj = adj[:,self.result]
        self.final_adj = self.trn_coo(self.before_adj,self.user_num, self.item_num)
        self.final_adj = self._make_torch_adj(self.final_adj).to(self.device)
        return self.final_adj

class NodeDrop(GeneralRecommender):
    """ Drop nodes in a graph.
        It is implemented by replace the embeddings of dropped nodes with random embeddings.
    """
    def __init__(self, config, dataset):
        super(NodeDrop, self).__init__(config, dataset)

    def forward(self, embeds, keep_rate):
        """
        :param embeds: the embedding matrix of nodes in the graph
        :param keep_rate: ratio of preserved nodes
        :return: the embeddings matrix after dropping nodes
        """
        if keep_rate == 1.0: return embeds
        node_num = self.n_users + self.n_items
        mask = (t.rand(node_num) + keep_rate).floor().view([-1, 1]).to(self.device)
        return embeds * mask

class New_NodeDrop(GeneralRecommender):
    """ Drop nodes in a graph.
        It is implemented by replace the embeddings of dropped nodes with random embeddings.
    """
    def __init__(self,config, dataset):
        super(New_NodeDrop, self).__init__(config, dataset)
        self.user_num = self.n_users
        self.item_num = self.n_items
        self.idx = New_idx(config, dataset)
        self.temp = 1

    def trn_coo(self,matrix,user_num,item_num):
        rows = matrix[0].cpu().numpy()
        cols = matrix[1].cpu().numpy()
        data = np.ones(rows.size)
        coo_matrix = sp.coo_matrix((data, (rows, cols)), shape=(user_num, item_num))
        return coo_matrix

    def _normalize_adj(self, mat):

        # Add epsilon to avoid divide by zero
        degree = np.array(mat.sum(axis=-1)) + 1e-10
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()

    def _make_torch_adj(self, mat):

        a = csr_matrix((self.user_num, self.user_num))
        b = csr_matrix((self.item_num, self.item_num))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        # mat = (mat + sp.eye(mat.shape[0])) * 1.0# MARK
        mat = self._normalize_adj(mat)
        # make torch tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).to(self.device)

    def forward(self, adj,keep_rate, augmentation):
        """
        :param embeds: the embedding matrix of nodes in the graph
        :param keep_rate: ratio of preserved nodes
        :return: the embeddings matrix after dropping nodes
        """
        self.result = self.idx.idx_sel(augmentation,self.temp,adj,keep_rate)
        self.before_adj = adj[:,self.result]
        self.final_adj = self.trn_coo(self.before_adj,self.user_num,self.item_num)
        self.final_adj = self._make_torch_adj(self.final_adj).to(self.device)

        return self.final_adj

class AdaptiveMask(nn.Module):
    """ Adaptively masking edges with learned weight (used in DCCF)
    """
    def __init__(self, head_list, tail_list, matrix_shape):
        """
        :param head_list: list of id about head nodes
        :param tail_list: list of id about tail nodes
        :param matrix_shape: shape of the matrix
        """
        super(AdaptiveMask, self).__init__()
        self.head_list = head_list
        self.tail_list = tail_list
        self.matrix_shape = matrix_shape

    def forward(self, head_embeds, tail_embeds):
        """
        :param head_embeds: embeddings of head nodes
        :param tail_embeds: embeddings of tail nodes
        :return: indices and values (representing an augmented graph in torch_sparse fashion)
        """
        head_embeddings = nn.functional.normalize(head_embeds)
        tail_embeddings = nn.functional.normalize(tail_embeds)
        edge_alpha = (t.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2
        A_tensor = SparseTensor(row=self.head_list, col=self.tail_list, value=edge_alpha, sparse_sizes=self.matrix_shape).cuda()
        D_scores_inv = (A_tensor.sum(dim=1).pow(-1)).clamp_(max=t.finfo(t.float32).max)
        G_indices = t.stack([self.head_list, self.tail_list], dim=0)
        G_values = D_scores_inv[self.head_list] * edge_alpha
        return G_indices, G_values

class AdaptiveMask_error(nn.Module):
    """ Adaptively masking edges with learned weight (used in DCCF)
    """
    def __init__(self, head_list, tail_list, matrix_shape):
        """
        :param head_list: list of id about head nodes
        :param tail_list: list of id about tail nodes
        :param matrix_shape: shape of the matrix
        """
        super(AdaptiveMask, self).__init__()
        self.head_list = head_list
        self.tail_list = tail_list
        self.matrix_shape = matrix_shape

    def forward(self, head_embeds, tail_embeds):
        """
        :param head_embeds: embeddings of head nodes
        :param tail_embeds: embeddings of tail nodes
        :return: indices and values (representing a augmented graph in torch_sparse fashion)
        """
        import torch_sparse
        head_embeddings = t.nn.functional.normalize(head_embeds)
        tail_embeddings = t.nn.functional.normalize(tail_embeds)
        edge_alpha = (t.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2
        A_tensor = torch_sparse.SparseTensor(row=self.head_list, col=self.tail_list, value=edge_alpha, sparse_sizes=self.matrix_shape).cuda()
        D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)
        G_indices = t.stack([self.head_list, self.tail_list], dim=0)
        G_values = D_scores_inv[self.head_list] * edge_alpha
        return G_indices, G_values

class SvdDecomposition(nn.Module):
    """ Utilize SVD to decompose matrix (used in LightGCL)
    """
    def __init__(self, svd_q):
        super(SvdDecomposition, self).__init__()
        self.svd_q = svd_q

    def forward(self, adj):
        """
        :param adj: torch sparse matrix
        :return: matrices obtained by SVD decomposition
        """
        svd_u, s, svd_v = t.svd_lowrank(adj, q=self.svd_q)
        u_mul_s = svd_u @ t.diag(s)
        v_mul_s = svd_v @ t.diag(s)
        del s
        return svd_u.T, svd_v.T, u_mul_s, v_mul_s

"""
Feature-based Augmentation
"""
class EmbedDrop(nn.Module):
    """ Drop embeddings by nn.Dropout
    """
    def __init__(self, p=0.2):
        super(EdgeDrop, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, embeds):
        """
        :param embeds: embedding matrix
        :return: embedding matrix after dropping
        """
        embeds = self.dropout(embeds)
        return embeds

class EmbedPerturb(nn.Module):
    """ Perturb embeddings
    """
    def __init__(self, eps):
        super(EmbedPerturb, self).__init__()
        self.eps = eps

    def forward(self, embeds):
        """ Perturbing embeddings with noise
        :param embeds: embedding matrix
        :return: perturbed embedding matrix
        """
        noise = (F.normalize(t.rand(embeds.shape).cuda(), p=2) * t.sign(embeds)) * self.eps
        embeds = embeds + noise
        return embeds

class KMeansClustering(nn.Module):
    """ Use KMeans to calculate cluster centers of embeddings (used in NCL)
    """
    def __init__(self, cluster_num, embedding_size):
        super(KMeansClustering, self).__init__()
        self.cluster_num = cluster_num
        self.embedding_size = embedding_size

    def forward(self, embeds):
        """
        :param embeds: embedding matrix
        :return: cluster information obtained by KMeans
        """
        centroids = t.rand([self.cluster_num, self.embedding_size]).cuda()
        ones = t.ones([embeds.shape[0], 1]).cuda()
        for i in range(1000):
            dists = (embeds.view([-1, 1, self.embedding_size]) - centroids.view([1, -1, self.embedding_size])).square().sum(-1)
            _, idxs = t.min(dists, dim=1)
            newCents = t.zeros_like(centroids)
            newCents.index_add_(0, idxs, embeds)
            clustNums = t.zeros([centroids.shape[0], 1]).cuda()
            clustNums.index_add_(0, idxs, ones)
            centroids = newCents / (clustNums + 1e-6)
        return centroids, idxs, clustNums
