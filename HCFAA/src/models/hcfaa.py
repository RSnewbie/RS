import torch as t
import Hyper_construct
from torch import nn
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from common.aug_utils import EdgeDrop, NodeDrop, New_EdgeDrop, New_NodeDrop
from common.abstract_recommender import GeneralRecommender
from common.loss_cf import cal_bpr_loss, reg_params, cal_infonce_loss
#from common.loss import BPRLoss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class HCFAA(GeneralRecommender):
	def __init__(self, config, dataset):
		super(HCFAA, self).__init__(config, dataset)

		self.augmentation = config['augmentation']
		self.cl_weight = config['cl_weight']
		self.temperature = config['temperature']
		self.keep_rate = config['keep_rate']
		self.layer_num = config['n_layers']
		self.reg_weight = config['reg_weight']
		self.emb_size = config['embed_size']
		self.hy_num = config['hyper_num']

		self.idx_adj = dataset.inter_matrix(form='coo').tocoo()
		self.idx_adj = np.mat([self.idx_adj.row, self.idx_adj.col])
		self.idx_adj = t.as_tensor(self.idx_adj,dtype=t.int64)

		self.n_nodes = self.n_users + self.n_items
		self.adj= self.get_norm_adj_mat(dataset.inter_matrix(form='coo').astype(np.float32)).to(self.device)
		self.u_adj,self.i_adj = Hyper_construct.H(dataset.inter_matrix(form='coo').astype(np.float32).toarray())
		self.G_u = t.from_numpy(self.u_adj).float().to(self.device)
		self.G_i = t.from_numpy(self.i_adj).float().to(self.device)
		self.hy_user_embeds = nn.Parameter(nn.init.xavier_uniform_(t.empty(self.n_users, self.emb_size)))
		self.hy_item_embeds = nn.Parameter(nn.init.xavier_uniform_(t.empty(self.n_items, self.emb_size)))
		self.user_embeds = nn.Parameter(nn.init.xavier_uniform_(t.empty(self.n_users, self.hy_num)))
		self.item_embeds = nn.Parameter(nn.init.xavier_uniform_(t.empty(self.n_items, self.hy_num)))

		self.edge_dropper = EdgeDrop()

		self.weight=nn.ParameterDict()

		for i in range(self.layer_num):
			self.weight['layer_%d' % (i + 1)] = nn.Parameter(t.empty(config['embedding_size'], config['embedding_size']))
			nn.init.xavier_uniform_(self.weight['layer_%d' % (i + 1)])  # 使用 xavier_normal_ 初始化权重矩阵

		if self.augmentation=='edge_drop':
			self.new_edge_dropper = New_EdgeDrop(config, dataset)
			self.new_pr = dataset.adtpr(self.idx_adj)
		elif self.augmentation=='random_walk':
			self.new_edge_dropper = New_EdgeDrop(config, dataset)
			self.new_cr = dataset.centrality(self.idx_adj)
		else:
			self.node_dropper = NodeDrop(config, dataset)
			self.new_node_dropper = New_NodeDrop(config, dataset)

	def get_norm_adj_mat(self, interaction_matrix):
		A = sp.dok_matrix((self.n_users + self.n_items,self.n_users + self.n_items), dtype=np.float32)
		inter_M = interaction_matrix
		inter_M_t = interaction_matrix.transpose()
		data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),[1] * inter_M.nnz))
		data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),[1] * inter_M_t.nnz)))
		A._update(data_dict)
		# norm adj matrix
		sumArr = (A > 0).sum(axis=1)
 		# add epsilon to avoid Devide by zero Warning
		diag = np.array(sumArr.flatten())[0] + 1e-7
		diag = np.power(diag, -0.5)
		D = sp.diags(diag)
		L = D * A * D
		# covert norm_adj matrix to tensor
		L = sp.coo_matrix(L)
		row = L.row
		col = L.col
		i = t.LongTensor(np.array([row, col]))
		data = t.FloatTensor(L.data)

		return t.sparse.FloatTensor(i, data, t.Size((self.n_nodes, self.n_nodes)))

	def _propagate(self, adj, embeds):
		return t.spmm(adj, embeds)

	def forward(self, adj, keep_rate):
		embeds = t.cat([self.user_embeds, self.item_embeds], axis=0)
		if self.augmentation == 'node_drop':
			embeds = self.node_dropper(embeds, keep_rate)
		embeds_list = [embeds]
		if self.augmentation == 'edge_drop':
			adj = self.edge_dropper(adj, keep_rate)
		for i in range(self.layer_num):
			random_walk = self.augmentation == 'random_walk'
			tem_adj = adj if not random_walk else self.edge_dropper(adj, keep_rate)
			embeds = self._propagate(adj, embeds_list[-1])
			embeds_list.append(embeds)
		embeds = sum(embeds_list)# / len(embeds_list)
		self.final_embeds = embeds
		return embeds[:self.n_users], embeds[self.n_users:]

	def  _forward(self, adj, keep_rate):
		embeds = t.cat([self.user_embeds, self.item_embeds], axis=0)
		if self.augmentation == 'node_drop':
			self.final_idx_adj = self.new_node_dropper(self.idx_adj,keep_rate,self.augmentation)
			embeds = self._propagate(self.final_idx_adj, embeds)
		embeds_list = [embeds]
		if self.augmentation == 'edge_drop':
			adj = self.new_edge_dropper(self.idx_adj,self.new_pr, keep_rate, self.augmentation)
		for i in range(self.layer_num):
			random_walk = self.augmentation == 'random_walk'
			tem_adj = adj if not random_walk else self.new_edge_dropper(self.idx_adj,self.new_cr, keep_rate, self.augmentation)
			embeds = self._propagate(tem_adj, embeds_list[-1])
			embeds_list.append(embeds)
		embeds = sum(embeds_list)# / len(embeds_list)
		self.final_embeds = embeds
		return embeds[:self.n_users], embeds[self.n_users:]

	def _pick_embeds(self, user_embeds, item_embeds, interaction):
		ancs = interaction[0]
		poss = interaction[1]
		negs = interaction[2]
		anc_embeds = user_embeds[ancs]
		pos_embeds = item_embeds[poss]
		neg_embeds = item_embeds[negs]
		return anc_embeds, pos_embeds, neg_embeds

	def hy_gcn(self, u_adj, i_adj):

		all_user_embeddings = []
		all_item_embeddings = []
		user_embeddings = self.hy_user_embeds
		item_embeddings = self.hy_item_embeds

		for i in range(self.layer_num):
			new_user_embeddings = t.mm(u_adj, user_embeddings)
			new_item_embeddings = t.mm(i_adj, item_embeddings)

			user_embeddings = F.leaky_relu(t.mm(new_user_embeddings, self.weight['layer_%d' %(i+1)]) + user_embeddings)
			item_embeddings = F.leaky_relu(t.mm(new_item_embeddings, self.weight['layer_%d' %(i+1)]) + item_embeddings)

			#user_embeddings = F.dropout(user_embeddings, p=0.2)
			#item_embeddings = F.dropout(item_embeddings, p=0.2)

			user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
			item_embeddings = F.normalize(item_embeddings, p=2, dim=1)

			all_user_embeddings.append(user_embeddings)
			all_item_embeddings.append(item_embeddings)

		hy_final_user_embeddings = t.stack(all_user_embeddings, dim=1).mean(dim=1)
		hy_final_item_embeddings = t.stack(all_item_embeddings, dim=1).mean(dim=1)

		return hy_final_user_embeddings, hy_final_item_embeddings

	def calculate_loss(self, interaction):

		user_embeds1, item_embeds1 = self.forward(self.adj, self.keep_rate)
		user_embeds2, item_embeds2 = self._forward(self.adj, self.keep_rate)
		user_embeds3, item_embeds3 = self.forward(self.adj, 1.0)
		user_embeds4, item_embeds4 = self.hy_gcn(self.G_u , self.G_i)

		anc_embeds1, pos_embeds1, neg_embeds1 = self._pick_embeds(user_embeds1, item_embeds1, interaction)
		anc_embeds2, pos_embeds2, neg_embeds2 = self._pick_embeds(user_embeds2, item_embeds2, interaction)
		anc_embeds3, pos_embeds3, neg_embeds3 = self._pick_embeds(user_embeds3, item_embeds3, interaction)
		anc_embeds4, pos_embeds4, neg_embeds4 = self._pick_embeds(user_embeds4, item_embeds4, interaction)

		bpr_loss = cal_bpr_loss(anc_embeds3, pos_embeds3, neg_embeds3) / anc_embeds3.shape[0]
		hy_bpr_loss = cal_bpr_loss(anc_embeds4, pos_embeds4, neg_embeds4) / anc_embeds4.shape[0]

		cl_loss = cal_infonce_loss(anc_embeds1, anc_embeds2, user_embeds2, self.temperature) + \
				  cal_infonce_loss(pos_embeds1, pos_embeds2, item_embeds2, self.temperature) + \
				  cal_infonce_loss(neg_embeds1, neg_embeds2, item_embeds2, self.temperature)
		cl_loss /= anc_embeds1.shape[0]

		reg_loss = self.reg_weight * reg_params(self)
		cl_loss *= self.cl_weight
		loss = hy_bpr_loss + bpr_loss + reg_loss + cl_loss

		return loss

	def full_sort_predict(self, interaction):
		user = interaction[0]

		user_embeddings, item_embeddings = self.forward(self.adj, 1.0)

		hy_u_emb, hy_i_emb = self.hy_gcn(self.G_u , self.G_i)
		user_embeddings = t.cat([user_embeddings , hy_u_emb],dim=1)
		item_embeddings = t.cat([item_embeddings , hy_i_emb],dim=1)

		user_e = user_embeddings[user, :]
		all_item_e = item_embeddings
		score = t.matmul(user_e, all_item_e.transpose(0, 1))

		return score
