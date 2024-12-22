import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F
import scipy.sparse as sp
from scipy import sparse
from torch.nn.parameter import Parameter
from scipy.sparse import coo_matrix,hstack

def H(adj):
    A = adj
    H_u = sparse.csr_matrix(A)

    D_u_v = np.array(H_u.sum(axis=1)).reshape(1, -1)
    D_u_v[D_u_v < 1] = 1e-5
    D_u_e = np.array(H_u.sum(axis=0)).reshape(1, -1)
    D_u_e[D_u_e < 1] = 1e-5
    temp1 = (H_u.transpose().multiply(np.sqrt(1.0 / D_u_v))).transpose()
    temp2 = temp1.transpose()
    A_u = temp1.multiply(1.0 / D_u_e).dot(temp2)
    A_u_save = A_u
    A_u = A_u.todense()

    A_ = adj.transpose()
    H_i = sparse.csr_matrix(A_)

    D_i_v = np.array(H_i.sum(axis=1)).reshape(1, -1)
    D_i_v[D_i_v < 1] = 1e-5
    D_i_e = np.array(H_i.sum(axis=0)).reshape(1, -1)
    D_i_e[D_i_e < 1] = 1e-5
    temp1 = (H_i.transpose().multiply(np.sqrt(1.0 / D_i_v))).transpose()
    temp2 = temp1.transpose()
    A_i = temp1.multiply(1.0 / D_i_e).dot(temp2)
    A_i_save = A_i
    A_i = A_i.todense()

    return A_u, A_i