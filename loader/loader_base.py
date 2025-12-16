from cgi import test
import os, torch
from ssl import PROTOCOL_TLS_CLIENT
import scipy.sparse as sp
import numpy as np
import pandas as pd
import torch.utils.data as data

class DataLoaderBase(data.Dataset):

    def __init__(self, args, logging):
        self.args = args
        self.data_name = args.dataset
        self.device=args.device
        self.data_dir = os.path.join(args.data_path, args.dataset)
        self.num_ng=1
        
        self.train_file = os.path.join(self.data_dir, 'train_list.npy')
        self.valid_file = os.path.join(self.data_dir, 'valid_list.npy')
        self.test_file = os.path.join(self.data_dir, 'test_list.npy')
        self.social_file = os.path.join(self.data_dir, 'social_list.npy')
        
        self.cf_train_data, self.train_user_dict = self.load_data(self.train_file)
        self.cf_valid_data, self.valid_user_dict = self.load_data(self.valid_file)
        self.cf_test_data, self.test_user_dict = self.load_data(self.test_file)
        self.social_data, self.social_dict = self.load_data(self.social_file)

        self.statistic_cf()
        
        self.train_user_pos_set = {int(u): set(map(int, items)) for u, items in self.train_user_dict.items()}
        # 블랙리스트 매핑 구축
        self.neg_upper, self.neg_map = self._build_neg_sampler(
            n_items=self.n_items,
            n_users=self.n_users,
            user_pos_set=self.train_user_pos_set
        )


    # DataLoaderBase 내부에 추가
    def _build_neg_sampler(self, n_items, n_users, user_pos_set):
        neg_upper = np.zeros(n_users, dtype=np.int64)
        neg_map = [dict() for _ in range(n_users)]
        for u in range(n_users):
            pos = set(map(int, user_pos_set.get(u, [])))
            L = n_items - len(pos)
            neg_upper[u] = L
            if L <= 0: 
                continue
            head_black = [x for x in pos if x < L]
            tail_white = iter(set(range(L, n_items)) - pos)
            for b in head_black:
                neg_map[u][b] = next(tail_white)
        return neg_upper, neg_map


    def ng_sample(self, u: int) -> int:
        """블랙리스트 매핑 기반 O(1) 네거티브 샘플링"""
        L = int(self.neg_upper[u])              # = n_items - |P_u|
        if L <= 0:
            return -1                           # 예외 사용자: 모든 아이템이 양성
        r = int(torch.randint(0, L, (1,)).item())  # worker별 시드로 결정됨
        return self.neg_map[u].get(r, r)

    def __len__(self):
        return len(self.cf_train_data[0])
        return self.num_ng * len(self.cf_train_data[0]) 
    
    def __getitem__(self,idx):
        # train_fill에 의존하지 않고 바로 (user, pos) 반환
        user = int(self.cf_train_data[0][idx])
        item_i = int(self.cf_train_data[1][idx])
        item_j = self.ng_sample(user)
        return user, item_i, item_j 
        user = int(self.train_fill[idx, 0])
        item_i = int(self.train_fill[idx, 1])
        item_j = int(self.train_fill[idx, 2])
        return user, item_i, item_j 
    
    def load_data(self, filename):    

        train_list = np.load(filename, allow_pickle=True)

        user = train_list[:,0]
        item = train_list[:,1]
        user_dict = dict()

        for uid, iid in train_list:
            if uid not in user_dict:
                user_dict[uid] = []
            user_dict[uid].append(iid)
        return (user, item), user_dict
 
    def statistic_cf(self):
        a=[max(self.cf_train_data[0]), max(self.cf_test_data[0]),max(self.cf_valid_data[0]),max(self.social_data[0]),max(self.social_data[1])]
        b=[max(self.cf_train_data[1]), max(self.cf_test_data[1]),max(self.cf_valid_data[1])]
        self.n_users = max(a) + 1
        self.n_items = max(b) + 1
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_valid = len(self.cf_valid_data[0])
        self.n_cf_test = len(self.cf_test_data[0])



    

    

