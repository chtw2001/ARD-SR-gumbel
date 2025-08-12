import numpy as np
import pandas as pd
import scipy.sparse as sp
import time
import math
from torch.utils.data import Dataset

from .loader_base import DataLoaderBase
from scipy.sparse import coo_matrix, csr_matrix
import os

class SRDataLoader(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        self.device=self.args.device
        self.train_batch_size = args.batch_size
        self.test_batch_size = args.batch_size
        self.train_h_list=list(self.cf_train_data[0])
        self.train_t_list=list(self.cf_train_data[1])

        self.train_social_h_list=list(self.social_data[0])
        self.train_social_t_list=list(self.social_data[1])
        self.social_graph,self.social_norm,_=self.buildSocialAdjacency()
        self.rating_mat, self.rating_mat_norm = self.getRatingAdjacency()

        self.train_item_dict=self.create_item_dict()
        self.print_info(logging)

    def buildSocialAdjacency(self):
        social_dict=dict()
        for ua,ub in zip(self.train_social_h_list,self.train_social_t_list):
            if ua not in social_dict:
                social_dict[ua] = []
            social_dict[ua].append(ub)

        row, col,entries, norm_entries = [], [],[], []
        train_h_list,train_t_list = self.train_social_h_list,self.train_social_t_list

        for i in range(len(train_h_list)):
            user=train_h_list[i]
            item=train_t_list[i]
            row += [user]
            col += [item]
            entries+=[1]
            if item in social_dict.keys():
                div=len(social_dict[item])
            else:
                div=1
            norm_entries += [1 / math.sqrt(len(social_dict[user])) /
            math.sqrt(div)]
        entries=np.array(entries)
        norm_entries=np.array(norm_entries)
        user=np.array(row)
        item=np.array(col)

        adj = coo_matrix((entries, (user, item)),shape=(self.n_users,self.n_users))
        norm_adj = coo_matrix((norm_entries, (user, item)),shape=(self.n_users, self.n_users))

        return  adj,norm_adj,social_dict

    def getRatingAdjacency(self):
        try:
            t1=time.time()
            inter_graph = sp.load_npz(self.data_dir + '/inter_adj_both.npz')
            inter_norm = sp.load_npz(self.data_dir + '/inter_norm_both.npz')
            print('already load adj matrix', inter_graph.shape, time.time() - t1)

        except Exception:
            self.train_item_dict=self.create_item_dict()
            inter_graph,inter_norm = self.buildRatingAdjacency()
            sp.save_npz(self.data_dir + '/inter_adj_both.npz', inter_graph)
            sp.save_npz(self.data_dir + '/inter_norm_both.npz', inter_norm)
        
        return inter_graph,inter_norm
    
    def buildRatingAdjacency(self):
        row, col, entries, norm_entries = [], [], [], []
        train_h_list,train_t_list = self.cf_train_data[0], self.cf_train_data[1]

        for i in range(len(train_h_list)):
            user=train_h_list[i]
            item=train_t_list[i]
            row += [user,item+self.n_users]
            col += [item+self.n_users,user]
            entries+=[1,1]
            degree=1 / math.sqrt(len(self.train_user_dict[user])) /math.sqrt(len(self.train_item_dict[item]))
            norm_entries += [degree,degree]
        entries=np.array(entries)
        user=np.array(row)
        item=np.array(col)

        adj = coo_matrix((entries, (user, item)),shape=(self.n_users+self.n_items,self.n_users+self.n_items))
        norm_adj = coo_matrix((norm_entries, (user, item)),shape=(self.n_users+self.n_items, self.n_users+self.n_items))

        return adj, norm_adj

    def create_item_dict(self):
        item_dict={}
        for i,j in enumerate(self.cf_train_data[0]):
            if self.cf_train_data[1][i] in item_dict.keys():
                item_dict[self.cf_train_data[1][i]].append(j)
            else:
                item_dict[self.cf_train_data[1][i]]=[j]
        return item_dict
    
    def print_info(self, logging):
        logging.info('n_users:     %d' % self.n_users)
        logging.info('n_items:     %d' % self.n_items)
        logging.info('n_cf_train:  %d' % self.n_cf_train)
        logging.info('n_cf_test:   %d' % self.n_cf_test)


    def getUserPosItems(self, user_batch):
        """
        user_batch: torch.LongTensor로, 여러 user ID를 담고 있음.
        return: list of lists
                user_batch의 각 user u에 대해, 
                self.train_user_dict[u]에서 item 리스트를 가져온 뒤 2D list로 반환
        """
        all_pos_items = []
        # user_batch가 LongTensor이므로, .item() 또는 int(u) 변환을 통해
        # 각 user ID를 뽑아내고, train_user_dict에서 아이템 리스트를 가져온다.
        for u in user_batch:
            user_id = int(u)  # 또는 u.item()
            # train_user_dict[user_id]가 해당 user가 interacted한 아이템 리스트
            if user_id in self.train_user_dict:
                all_pos_items.append(self.train_user_dict[user_id])
            else:
                # 혹시라도 딕셔너리에 없으면 빈 리스트 반환
                all_pos_items.append([])
        return all_pos_items


class DataDiffusion(Dataset):
    def __init__(self, data):
        # Filter out zero rows and keep a mapping to original indices
        self.data = data
        self.non_zero_indices = []

        for i in range(self.data.shape[0]):
            if (self.data[i] != 0).sum():
                self.non_zero_indices.append(i)    

    def __getitem__(self, index):
        # Map the index to the corresponding non-zero row
        original_index = self.non_zero_indices[index]
        item = self.data[original_index]
        return original_index, item

    def __len__(self):
        return len(self.non_zero_indices)


class DataDiffusionCL(Dataset):
    def __init__(self, data, current_epoch, ce=None, max_epochs=50, initial_prop=0.4, seed=42):

        self.data = data
        self.non_zero_indices = []
        self.max_epochs = max_epochs
        self.current_epoch = current_epoch
        self.initial_prop = initial_prop
        self.seed = seed
        
        # 재현성을 위한 seed 설정
        np.random.seed(self.seed)
        
        non_zero_indices = [i for i, row in enumerate(self.data) if (row != 0).sum() > 0]  
        max_samples = len(non_zero_indices)  
        proportion = min(1, self.initial_prop + (1 - self.initial_prop) * self.current_epoch / self.max_epochs)
        num_samples = int(proportion * max_samples)

        row_degrees = np.array([(row == 1).sum() for row in self.data])
        row_degrees = row_degrees[non_zero_indices] 
        degree_ranks = np.argsort(np.argsort(-row_degrees))  
        if ce is not None:
            ce = ce[non_zero_indices]  
            ce_ranks = np.argsort(np.argsort(ce))  
        else:
            ce_ranks = np.zeros_like(degree_ranks)  
        aggregated_ranks =  degree_ranks + ce_ranks

        top_indices = np.argsort(aggregated_ranks)[:num_samples]
        for idx in top_indices:
            i = non_zero_indices[idx] 
            self.non_zero_indices.append(i)
        
    def __len__(self):
        return len(self.non_zero_indices)
    
    def __getitem__(self, index):
        original_index = self.non_zero_indices[index]
        item = self.data[original_index]
        return original_index, item
