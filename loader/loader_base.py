from cgi import test
import os
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
        
        # 재현성을 위한 seed 설정
        self.seed = args.seed
        np.random.seed(self.seed)

        self.train_file = os.path.join(self.data_dir, 'train_list.npy')
        self.valid_file = os.path.join(self.data_dir, 'valid_list.npy')
        self.test_file = os.path.join(self.data_dir, 'test_list.npy')
        self.social_file = os.path.join(self.data_dir, 'social_list.npy')
        
        self.cf_train_data, self.train_user_dict = self.load_data(self.train_file)
        self.cf_valid_data, self.valid_user_dict = self.load_data(self.valid_file)
        self.cf_test_data, self.test_user_dict = self.load_data(self.test_file)
        self.social_data, self.social_dict = self.load_data(self.social_file)

        self.statistic_cf()


    def ng_sample(self): 
        # 재현성을 위해 seed 재설정
        np.random.seed(self.seed)
        import random
        random.seed(self.seed)
        
        self.train_fill=[]
        for x in range(len(self.cf_train_data[0])):
            u, i = self.cf_train_data[0][x], self.cf_train_data[1][x]
            for t in range(self.num_ng):
                j = np.random.randint(self.n_items)
                while  j in self.train_user_dict[u]:
                    j = np.random.randint(self.n_items)
                self.train_fill.append([u, i, j])

    def __len__(self):     
   				 
        return self.num_ng * len(self.cf_train_data[0]) 
    
    def __getitem__(self,idx):
        user = self.train_fill[idx][0]
        item_i = self.train_fill[idx][1]
        item_j = self.train_fill[idx][2]
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
        a=[max(self.cf_train_data[0]), max(self.cf_test_data[0]),max(self.cf_valid_data[0]),
        max(self.social_data[0]),max(self.social_data[1])]
        b=[max(self.cf_train_data[1]), max(self.cf_test_data[1]),max(self.cf_valid_data[1])]
        self.n_users = max(a) + 1
        self.n_items = max(b) + 1
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_valid = len(self.cf_valid_data[0])
        self.n_cf_test = len(self.cf_test_data[0])


    

    

