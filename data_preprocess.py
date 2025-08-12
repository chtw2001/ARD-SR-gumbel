## data preprocessing for ciao
import os
import pandas as pd
import random
import numpy as np

# Set seed for reproducible preprocessing
random.seed(42)
np.random.seed(42)

os.chdir('/home/user/sh/data/ciao_with_rating_timestamp')  ## replace with home directory
import scipy.io as sio

rating = sio.loadmat('rating_with_timestamp.mat')
rating=rating['rating']
rating=pd.DataFrame(rating, columns = ['u','i','cid','rating','did','time'])
trust= sio.loadmat('trust.mat')
trust=trust['trust']
trust=pd.DataFrame(trust, columns = ['ua','ub'])
rating=rating.iloc[:,[0,1,2,3,5]].astype(int)
rating=rating.rename(columns={0:'u',1:'i',2:'cid',3:'rating',5:'time'})

# 4점 이상인 interaction만 남김
rating=rating.loc[rating['rating']>3].reset_index(drop=True)
summary_user=rating.groupby('u')['i'].count().reset_index()
summary_item=rating.groupby('i')['u'].count().reset_index()
# 3개 이상 상호작용한 user/item만 남김
all_user=summary_user[(summary_user.i >2 )].u
all_item=summary_item[summary_item.u >2].i
rating=rating[rating.u.isin(list(all_user))& rating.i.isin(list(all_item))].iloc[:,[0,1,3,4]].reset_index(drop=True)
trust=trust[trust.ua.isin(list(all_user)) & trust.ub.isin(list(all_user))].iloc[:,0:2].reset_index(drop=True)


rating['cnt'] = rating.groupby("u")["time"].transform('rank', method='first',ascending=False)
rating['ttl'] = rating.groupby("u")["i"].transform('count')
rating['ratio']=rating['cnt']/rating['ttl']

rating1=rating[rating.ttl>9] # 상호작용 10개 이상
rating2=rating[rating.ttl==2] # 2개
rating3=rating[(rating.ttl<=9) & (rating.ttl>2)] # 3~9개

train_rating1=rating1[rating1.ratio>0.2] # 상호작용 10개 이상 중 오래된 80%
test_valid_rating=rating1[ rating1.ratio<=0.2] # 상호작용 10개 이상 중 최산 20%

def generate_unique_random_row_numbers(group):
    num_rows = len(group)
    group['sub_idx'] = random.sample(range(1,num_rows+1), num_rows)
    return group

# 2개 상호작용한 user는 1개는 train, 1개는 test
rating2=rating2.groupby('u').apply(generate_unique_random_row_numbers)
train_rating2=rating2[rating2.sub_idx==1].iloc[:,[0,1,2,3]]
test_valid_rating2=rating2[rating2.sub_idx==2].iloc[:,[0,1,2,3]]

# 3~9개 상호작용한 user는 80%는 train, 20%는 test
rating3=rating3.groupby('u').apply(generate_unique_random_row_numbers)
train_rating3=rating3[rating3.sub_idx/rating3.ttl<=0.8].iloc[:,[0,1,2,3]]
test_valid_rating3=rating3[rating3.sub_idx/rating3.ttl>0.8].iloc[:,[0,1,2,3]]

train_rating=pd.concat([train_rating1.iloc[:,[0,1,2,3]],train_rating2,train_rating3])
test_valid_rating=pd.concat([test_valid_rating.iloc[:,[0,1,2,3]],test_valid_rating2,test_valid_rating3])

# test와 valid 분할
test_valid_rating=test_valid_rating.groupby('u').apply(generate_unique_random_row_numbers)
valid_rating=test_valid_rating[test_valid_rating.sub_idx==1].iloc[:,[0,1,2,3]]
test_rating=test_valid_rating[test_valid_rating.sub_idx==2].iloc[:,[0,1,2,3]]

# user, item id 재매핑
all_user=pd.concat([train_rating['u'],valid_rating['u'],test_rating['u']]).unique()
all_item=pd.concat([train_rating['i'],valid_rating['i'],test_rating['i']]).unique()

user_id_map={}
for i,user in enumerate(all_user):
    user_id_map[user]=i

item_id_map={}
for i,item in enumerate(all_item):
    item_id_map[item]=i

train_rating['u']=train_rating['u'].map(user_id_map)
train_rating['i']=train_rating['i'].map(item_id_map)
valid_rating['u']=valid_rating['u'].map(user_id_map)
valid_rating['i']=valid_rating['i'].map(item_id_map)
test_rating['u']=test_rating['u'].map(user_id_map)
test_rating['i']=test_rating['i'].map(item_id_map)

trust['ua']=trust['ua'].map(user_id_map)
trust['ub']=trust['ub'].map(user_id_map)

# npy 파일로 저장
train_list=train_rating.iloc[:,[0,1]].values
valid_list=valid_rating.iloc[:,[0,1]].values
test_list=test_rating.iloc[:,[0,1]].values
social_list=trust.iloc[:,[0,1]].values

np.save('train_list.npy',train_list)
np.save('valid_list.npy',valid_list)
np.save('test_list.npy',test_list)
np.save('social_list.npy',social_list)

print('train_list.shape:',train_list.shape)
print('valid_list.shape:',valid_list.shape)
print('test_list.shape:',test_list.shape)
print('social_list.shape:',social_list.shape)
print('n_users:',len(all_user))
print('n_items:',len(all_item))