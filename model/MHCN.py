

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

def to_tensor(coo_mat,args):
    values = coo_mat.data
    indices = np.vstack((coo_mat.row, coo_mat.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo_mat.shape
    tensor_sparse=torch.sparse.FloatTensor(i, v, torch.Size(shape))
    tensor_sparse=tensor_sparse.to(args.device)
    return tensor_sparse

def permute_sparse(input, dims):
    dims = torch.LongTensor(dims)
    return torch.sparse_coo_tensor(indices=input._indices()[dims]
                                   , values=input._values()
                                   , size=torch.Size(torch.tensor(input.size())[dims]))

class MHCN(nn.Module):
    def __init__(self,data,args):
        super(MHCN, self).__init__()
        self.config = args
        self.n_layers = self.config.n_layers
        self.emb_size = self.config.embed_dim
        self.data=data

        self.num_users, self.num_items= self.data.n_users,self.data.n_items

        # 재현성을 위한 seed 설정
        if hasattr(args, 'seed'):
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

        self.init_channel()

        self.weights = torch.nn.ParameterDict()
        self.n_channel = 4
        
        for i in range(self.n_channel):
            name = "gating%d" % (i + 1)
            self.weights[name] = nn.Parameter(torch.Tensor(self.emb_size, self.emb_size)) 
            nn.init.xavier_uniform_(self.weights[name])

            name = "gating_bias%d" % (i + 1)
            self.weights[name] = nn.Parameter(torch.Tensor(1, self.emb_size)) 
            nn.init.xavier_uniform_(self.weights[name])

            name = "sgating%d" % (i + 1)
            self.weights[name] = nn.Parameter(torch.Tensor(self.emb_size, self.emb_size)) 
            nn.init.xavier_uniform_(self.weights[name])

            name = "sgating_bias%d" % (i + 1)
            self.weights[name] = nn.Parameter(torch.Tensor(1, self.emb_size)) 
            nn.init.xavier_uniform_(self.weights[name])

        self.weights["attention"] = nn.Parameter(torch.Tensor(1, self.emb_size)) 
        nn.init.xavier_uniform_(self.weights["attention"])

        self.weights["attention_mat"] = nn.Parameter(torch.Tensor(self.emb_size, self.emb_size)) 
        nn.init.xavier_uniform_(self.weights["attention_mat"])

        self.user_embeddings = nn.Embedding(self.num_users,self.emb_size)
        self.item_embeddings = nn.Embedding(self.num_items,self.emb_size)

        nn.init.normal_(self.user_embeddings.weight, std=0.1)
        nn.init.normal_(self.item_embeddings.weight, std=0.1)

    def init_channel(self):
    # user-user
        self.H_s, self.H_j, self.H_p = self.buildMotifInducedAdjacencyMatrix()
        self.H_s = to_tensor(self.H_s,self.config)
        self.H_j = to_tensor(self.H_j,self.config)
        self.H_p = to_tensor(self.H_p,self.config)
        R=self.buildJointAdjacency()
        self.R = to_tensor(R,self.config)
        
    def buildSparseRelationMatrix(self):
        row, col, entries = [], [], []
        for i in range(len(self.data.train_social_h_list)):
        #for pair in self.social.relation:
            # symmetric matrix
            row += [self.data.train_social_h_list[i]]
            col += [self.data.train_social_t_list[i]]
            entries += [1.0]
        AdjacencyMatrix = coo_matrix(
            (entries, (row, col)),
            shape=(self.num_users, self.num_users),
            dtype=np.float32)
        return AdjacencyMatrix

    # user-item
    def buildSparseRatingMatrix(self):
        row, col, entries = [], [], []
        for i in range(len(self.data.train_h_list)):
        #for pair in self.social.relation:
            # symmetric matrix
            row += [self.data.train_h_list[i]]
            col += [self.data.train_t_list[i]]
            entries += [1.0]
        ratingMatrix = coo_matrix(
            (entries, (row, col)),
            shape=(self.num_users, self.num_items),
            dtype=np.float32)
        return ratingMatrix

    def buildJointAdjacency(self):
        row, col, entries = [], [], []
        for i in range(len(self.data.train_h_list)):
        #for pair in self.social.relation:
            # symmetric matrix
            user=self.data.train_h_list[i]
            item=self.data.train_t_list[i]
            row += [user]
            col += [item]
            entries += [1 / math.sqrt(len(self.data.train_user_dict[user])) /
            math.sqrt(len(self.data.train_item_dict[item]))]
        entries=np.array(entries)
        user=np.array(row)
        item=np.array(col)

        norm_adj = coo_matrix(
            (entries, (user, item)),
            shape=(self.num_users, self.num_items))

        return norm_adj

    def buildMotifInducedAdjacencyMatrix(self):
        # social graph, user-user
        S = self.buildSparseRelationMatrix()
        Y = self.buildSparseRatingMatrix()
        self.userAdjacency = Y.tocsr()
        self.itemAdjacency = Y.T.tocsr()
        B = S.multiply(S.T)
        U = S - B
        C1 = (U.dot(U)).multiply(U.T)
        A1 = C1 + C1.T
        C2 = (B.dot(U)).multiply(U.T) + (U.dot(B)).multiply(U.T) + (
            U.dot(U)).multiply(B)
        A2 = C2 + C2.T
        C3 = (B.dot(B)).multiply(U) + (B.dot(U)).multiply(B) + (U.dot(B)
                                                                ).multiply(B)
        A3 = C3 + C3.T
        A4 = (B.dot(B)).multiply(B)
        C5 = (U.dot(U)).multiply(U) + (U.dot(U.T)).multiply(U) + (U.T.dot(U)
                                                                  ).multiply(U)
        A5 = C5 + C5.T
        A6 = (U.dot(B)).multiply(U) + (B.dot(U.T)).multiply(U.T) + (
            U.T.dot(U)).multiply(B)
        A7 = (U.T.dot(B)).multiply(U.T) + (B.dot(U)).multiply(U) + (
            U.dot(U.T)).multiply(B)
        A8 = (Y.dot(Y.T)).multiply(B)
        A9 = (Y.dot(Y.T)).multiply(U)
        A9 = A9 + A9.T
        A10 = Y.dot(Y.T) - A8 - A9
        # addition and row-normalization
        H_s = sum([A1, A2, A3, A4, A5, A6, A7])
        H_s = H_s.multiply(1.0 / H_s.sum(axis=1).reshape(-1, 1))
        H_j = sum([A8, A9])
        H_j = H_j.multiply(1.0 / H_j.sum(axis=1).reshape(-1, 1))
        H_p = A10
        H_p = H_p.multiply(H_p > 1)
        H_p = H_p.multiply(1.0 / H_p.sum(axis=1).reshape(-1, 1))

        # return [H_s.toarray(), H_j.toarray(), H_p.toarray()]
        return [H_s, H_j, H_p]

    def self_gating(self, em, channel):
        """
        em: (num_users, emb_size)
        """
        return torch.mul(
            em,
            torch.sigmoid(
                torch.matmul(em, self.weights["gating%d" % channel]) +
                self.weights["gating_bias%d" % channel]))

    def self_supervised_gating(self, em, channel):
        return torch.mul(em,torch.sigmoid(torch.matmul(em, self.weights["sgating%d" % channel]) +
                self.weights["sgating_bias%d" % channel]))

    def channel_attention(self, *channel_embeddings):
        """
            channel_embeddings_1: (num_user, emb_size)
            attention_mat: (emb_size, emb_size)
            attention: (1, emb_size)
        """
        weights = []
        for embedding in channel_embeddings:
            # ((num_user, emb_size) * (emb_size, emb_size)) @ (1, emb_size) = (num_user, emb_size) @ (1, emb_size)
            # = (num_user, emb_size) -> (num_user, )
            weights.append(
                torch.sum(torch.mul(torch.matmul(embedding, self.weights[
                            "attention_mat"]), self.weights["attention"]), 1))
        t = torch.stack(weights)
        # (num_user, channel_num)
        score = F.softmax(torch.permute(t, (1, 0)))
        mixed_embeddings = 0.0
        for i in range(len(weights)):
            # (emb_size, num_user) @
            # (num_user, emb_size) @ (num_user, 1) -> (num_user, emb_size)
            mixed_embeddings += torch.permute(torch.mul(
                    torch.permute(channel_embeddings[i], (1, 0)),
                    torch.permute(score, (1, 0))[i]),(1, 0))
        return mixed_embeddings, score

    def infer_embedding(self):

        # self-gating
        user_embeddings_c1 = self.self_gating(self.user_embeddings.weight, 1)
        user_embeddings_c2 = self.self_gating(self.user_embeddings.weight, 2)
        user_embeddings_c3 = self.self_gating(self.user_embeddings.weight, 3)
        simple_user_embeddings = self.self_gating(self.user_embeddings.weight,4)
        all_embeddings_c1 = [user_embeddings_c1]
        all_embeddings_c2 = [user_embeddings_c2]
        all_embeddings_c3 = [user_embeddings_c3]
        all_embeddings_simple = [simple_user_embeddings]
        item_embeddings = self.item_embeddings.weight
        all_embeddings_i = [item_embeddings]

        # multi-channel convolution
        for k in range(self.n_layers):
            mixed_embedding = self.channel_attention(
                user_embeddings_c1, user_embeddings_c2,
                user_embeddings_c3)[0] + simple_user_embeddings / 2.0
            # Channel S
            user_embeddings_c1 = torch.matmul(self.H_s, user_embeddings_c1)
            norm_embeddings = F.normalize(user_embeddings_c1, dim=1, p=2)
            all_embeddings_c1 += [norm_embeddings]
            # Channel J
            user_embeddings_c2 = torch.matmul(self.H_j, user_embeddings_c2)
            norm_embeddings = F.normalize(user_embeddings_c2, dim=1, p=2)
            all_embeddings_c2 += [norm_embeddings]
            # Channel P
            user_embeddings_c3 = torch.matmul(self.H_p, user_embeddings_c3)
            norm_embeddings = F.normalize(user_embeddings_c3, dim=1, p=2)
            all_embeddings_c3 += [norm_embeddings]
            # item convolution
            new_item_embeddings = torch.matmul(
                    permute_sparse(self.R, (1, 0)), mixed_embedding)
            norm_embeddings = F.normalize(new_item_embeddings, dim=1, p=2)
            all_embeddings_i += [norm_embeddings]
            simple_user_embeddings = torch.matmul(self.R, item_embeddings)
            all_embeddings_simple += [F.normalize(simple_user_embeddings, dim=1, p=2)]
            item_embeddings = new_item_embeddings

        # averaging the channel-specific embeddings
        user_embeddings_c1 = torch.mean(torch.stack(all_embeddings_c1,dim=0),dim=0)
        user_embeddings_c2 = torch.mean(torch.stack(all_embeddings_c2,dim=0),dim=0)
        user_embeddings_c3 = torch.mean(torch.stack(all_embeddings_c3,dim=0),dim=0)
        simple_user_embeddings = torch.mean(torch.stack(all_embeddings_simple,dim=0), dim=0)
        item_embeddings = torch.mean(torch.stack(all_embeddings_i,dim=0), dim=0)

        # aggregating channel-specific embeddings
        final_item_embeddings = item_embeddings
        final_user_embeddings, attention_score = self.channel_attention(
            user_embeddings_c1, user_embeddings_c2, user_embeddings_c3)
        final_user_embeddings= final_user_embeddings +simple_user_embeddings 

        return final_user_embeddings, final_item_embeddings
    
    def hierarchical_self_supervision(self, em, adj):
        def row_shuffle(embedding):
            return embedding[torch.randperm(embedding.size()[0])]

        def row_column_shuffle(embedding):
            embedding = torch.permute(embedding, (1, 0))
            corrupted_embedding = torch.permute(
                embedding[torch.randperm(embedding.size()[0])],
                (1, 0))
            return corrupted_embedding[torch.randperm(
                corrupted_embedding.size()[0])]

        def score(x1, x2):
            return torch.sum(torch.multiply(x1, x2), dim=1)

        user_embeddings = em
        edge_embeddings = torch.matmul(adj, user_embeddings)

        # Local MIN
        pos = score(user_embeddings, edge_embeddings)
        neg1 = score(row_shuffle(user_embeddings), edge_embeddings)
        neg2 = score(row_column_shuffle(edge_embeddings), user_embeddings)
        local_loss = torch.sum(-torch.log(torch.sigmoid(pos - neg1)) -
                                torch.log(torch.sigmoid(neg1 - neg2)))
        local_loss=local_loss/user_embeddings.size()[0]

        # Global MIN
        graph = torch.mean(edge_embeddings, dim=0)
        pos = score(edge_embeddings, graph)
        neg1 = score(row_column_shuffle(edge_embeddings), graph)
        global_loss = torch.sum(-torch.log(torch.sigmoid(pos - neg1)))
        global_loss=global_loss/edge_embeddings.size()[0]

        return global_loss + local_loss
    
    
    def bpr_loss(self,u_idx, v_idx, neg_idx):   
        """
        u_idx, v_idx, neg_idx = inputs
        """
        #u_idx, v_idx, neg_idx = inputs

        final_user_embeddings, final_item_embeddings = self.infer_embedding()

        # create self-supervised loss
        ss_loss = 0.0
        ss_loss += self.hierarchical_self_supervision(
            self.self_supervised_gating(final_user_embeddings, 1), self.H_s)
        ss_loss += self.hierarchical_self_supervision(
            self.self_supervised_gating(final_user_embeddings, 2), self.H_j)
        ss_loss += self.hierarchical_self_supervision(
            self.self_supervised_gating(final_user_embeddings, 3), self.H_p)

        # embedding look-up
        user_emb = final_user_embeddings[u_idx,:]
        pos_emb = final_item_embeddings[v_idx,:]
        neg_emb = final_item_embeddings[neg_idx,:]

        pos_scores = torch.mul(user_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(user_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        user0=self.user_embeddings(u_idx)
        pos0=self.item_embeddings(v_idx)
        neg0=self.item_embeddings(neg_idx)

        reg_loss = (1/2)*(user0.norm(2).pow(2) + 
                         pos0.norm(2).pow(2)  +
                         neg0.norm(2).pow(2))/float(len(u_idx))

        return loss,reg_loss,ss_loss

    def forward(self,user_id,item_id):
        all_user,all_item=self.infer_embedding()
        user_emb=all_user[user_id]
        item_emb=all_item[item_id]

        cf_scores = torch.mul(user_emb, item_emb)
        cf_scores = torch.sum(cf_scores, dim=1)

        return cf_scores

    def getUsersRating(self,users):

        all_users, all_items = self.infer_embedding()
        user_embed=all_users[users]
        cf_scores = torch.mm(user_embed, all_items.t())

        return cf_scores
    
    def get_social_embed(self,graph,base):
        all_emb = base
        embs = [all_emb]
        for layer in range(3):
            all_emb = torch.sparse.mm(graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        return light_out

