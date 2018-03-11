import pandas as pd
import os

path = '../../data_ori/'

'''
AB榜测试集-evaluation_public.csv  483931
训练数据-ccf_first_round_shop_info.csv  shop_id 8477  mall_id 97
训练数据-ccf_first_round_user_shop_behavior.csv  1138015
'''
shop_info = pd.read_csv(path+'shop_info.csv', usecols=['shop_id', 'mall_id'])

train = pd.read_csv(path+'train.csv')
test = pd.read_csv('../../data_ori/test.csv')
train = train.drop_duplicates()

train = pd.merge(left=train, right=shop_info, how='left', on='shop_id')


for mall in train['mall_id'].unique():
    df_tr = train[train['mall_id']==mall]
    df_tr.to_csv('../../data/train/{}.csv'.format(mall, index=False))

    df_te = test[test['mall_id'] == mall]
    df_te.to_csv('../../data/test/{}.csv'.format(mall), index=False)


