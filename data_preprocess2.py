# preprocessing for douban, filmtrust dataset
import os
import numpy as np
import pandas as pd
import random

# Set seed for reproducible preprocessing
random.seed(42)
np.random.seed(42)

def preprocess_data(
    rating_file,
    social_file,
    rating_threshold,
    min_interaction,
    train_ratio,
    valid_ratio,
    test_ratio,
    save_dir
):
    """
    rating_file : user, item, rating (ex: u i r)
    social_file : userA, userB, weight (ex: ua ub 1)
    rating_threshold : rating이 이 값 이하인 것은 삭제
    min_interaction : user 또는 item이 이 값 이하의 인터랙션만 가진다면 삭제
    train_ratio, valid_ratio, test_ratio : 데이터 분할 비율 (합이 1)
    save_dir : 최종 npy 파일 저장 경로
    """
    
    # =======================
    # 1) Rating Data 로드
    # =======================
    # print("Loading interaction data from:", rating_file)

    rating_df = pd.read_csv(rating_file, sep='\t', header=None, names=['user','item','rating'])
    # rating 필터링
    if 'rating' in rating_df.columns and rating_df['rating'].notna().all():
        rating_df = rating_df[rating_df['rating'] >= rating_threshold].reset_index(drop=True)
    
    origin_user, origin_item, origin_interaction = rating_df['user'].nunique(), rating_df['item'].nunique(), len(rating_df)
   
    # =======================
    # 2) 유저/아이템별 인터랙션 수 카운팅
    # =======================
    # user가 사용한 item 수
    user_count = rating_df.groupby('user')['item'].count().reset_index(name='count_item')
    # item이 구매 된 횟수
    item_count = rating_df.groupby('item')['user'].count().reset_index(name='count_user')
    
    # interaction 필터링
    valid_users = user_count[user_count['count_item'] >= min_interaction]['user'].unique()
    valid_items = item_count[item_count['count_user'] >= min_interaction]['item'].unique()
    
    rating_df = rating_df[
        rating_df['user'].isin(valid_users) & 
        rating_df['item'].isin(valid_items)
    ].reset_index(drop=True)
    
    print("After filtering >= {} rating or users/items >= {} interactions:".format(rating_threshold, min_interaction))
    print("   # of interactions:", origin_interaction, '-->', len(rating_df))
    print("   # of users:", origin_user, '-->', rating_df['user'].nunique())
    print("   # of items:", origin_item, '-->', rating_df['item'].nunique())

    # =======================
    # 3) data split
    # =======================
    rating_df = rating_df.sample(frac=1, random_state=42).reset_index(drop=True)  # 셔플
    
    total_num = len(rating_df)
    train_end = int(total_num * train_ratio)
    valid_end = train_end + int(total_num * valid_ratio)
    
    train_data = rating_df.iloc[:train_end]
    valid_data = rating_df.iloc[train_end:valid_end]
    test_data  = rating_df.iloc[valid_end:]
    
    # =======================
    # 4) Social Data 로드
    # =======================
    # print("Loading social data from:", social_file)
    social_df = pd.read_csv(social_file, sep='\t', header=None, names=['ua','ub'])
    origin_social = len(social_df)
    
    # interaction graph에 남아있는 유저만 소셜 그래프에 반영
    final_users = set( train_data['user'].unique() ).union(
                  set( valid_data['user'].unique() ), 
                  set( test_data['user'].unique()) )
    
    social_df = social_df[
        social_df['ua'].isin(final_users) & 
        social_df['ub'].isin(final_users)
    ].reset_index(drop=True)
    
    print("After social filtering, # of edges:", origin_social, '-->', len(social_df))
    
    # =======================
    # 5) 유저/아이템 인덱스 매핑
    # =======================
    #   - user / item id를 0,1,2,... 로 factorize
    #   - social_df에서도 동일한 user mapping 적용
    # =======================
    unique_user = sorted(list(final_users))  # 최종 user set
    unique_item = sorted(list(pd.concat([train_data['item'], valid_data['item'], test_data['item']]).unique()))
    
    def map_data(df):
        user_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_user)}
        item_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_item)}
        
        df['user'] = df['user'].map(user_mapping)
        df['item'] = df['item'].map(item_mapping)
        return df
    
    train_data = map_data(train_data)
    valid_data = map_data(valid_data)
    test_data = map_data(test_data)
    
    # social data도 동일한 user mapping 적용
    user_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_user)}
    social_df['ua'] = social_df['ua'].map(user_mapping)
    social_df['ub'] = social_df['ub'].map(user_mapping)
    
    # =======================
    # 6) 최종 데이터 저장
    # =======================
    def save_txt(path, data):
        data.to_csv(path, sep='\t', header=False, index=False)
    
    # txt 파일로 저장
    save_txt(os.path.join(save_dir, 'train.txt'), train_data[['user', 'item']])
    save_txt(os.path.join(save_dir, 'valid.txt'), valid_data[['user', 'item']])
    save_txt(os.path.join(save_dir, 'test.txt'), test_data[['user', 'item']])
    save_txt(os.path.join(save_dir, 'social.txt'), social_df[['ua', 'ub']])
    
    # npy 파일로 저장
    train_list = train_data[['user', 'item']].values
    valid_list = valid_data[['user', 'item']].values
    test_list = test_data[['user', 'item']].values
    social_list = social_df[['ua', 'ub']].values
    
    np.save(os.path.join(save_dir, 'train_list.npy'), train_list)
    np.save(os.path.join(save_dir, 'valid_list.npy'), valid_list)
    np.save(os.path.join(save_dir, 'test_list.npy'), test_list)
    np.save(os.path.join(save_dir, 'social_list.npy'), social_list)
    
    print("Final statistics:")
    print("   # of users:", len(unique_user))
    print("   # of items:", len(unique_item))
    print("   # of train interactions:", len(train_list))
    print("   # of valid interactions:", len(valid_list))
    print("   # of test interactions:", len(test_list))
    print("   # of social edges:", len(social_list))
    print("Files saved to:", save_dir)
