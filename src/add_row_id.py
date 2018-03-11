from datetime import datetime

from feat_engineering import *
import pandas as pd
import numpy as np

path = '../output/'

df_xgb = pd.read_csv(path + 'xgb_prob.csv')
df_lgb = pd.read_csv(path + 'lgb_prob.csv')
df_ovr = pd.read_csv(path + 'ovr_prob.csv')
df_nn = pd.read_csv(path + 'nn_prob.csv')
df_rf = pd.read_csv(path + 'rf_prob.csv')
df_et = pd.read_csv(path + 'et_prob.csv')

shop_info = pd.read_csv('../data_ori/shop_info.csv', usecols=['shop_id', 'mall_id'])
train_all = pd.read_csv(path + '../data_ori/train.csv', usecols=['shop_id'])
train_all = pd.merge(left=train_all, right=shop_info, how='left', on='shop_id')
test_all = pd.read_csv('../data_ori/test.csv', usecols=['row_id', 'mall_id'])

df = pd.DataFrame()
df_result = pd.DataFrame()
last_index = 0
for t, mall in enumerate(train_all['mall_id'].unique(), start=1):
    train = train_all[train_all['mall_id'] == mall].reset_index(drop=True)
    train_all = train_all[train_all['mall_id'] != mall]
    test = test_all[test_all['mall_id'] == mall].reset_index(drop=True)

    cur_index = last_index + len(test_all[test_all['mall_id'] == mall])
    xgb = df_xgb.iloc[last_index:cur_index]
    lgb = np.array(df_lgb)[last_index:cur_index]
    ovr = np.array(df_ovr)[last_index:cur_index]
    nn = np.array(df_nn)[last_index:cur_index]
    rf = np.array(df_rf)[last_index:cur_index]
    et = np.array(df_et)[last_index:cur_index]
    last_index = cur_index

    df_tmp = pd.DataFrame(et)
    df_tmp['row_id'] = test['row_id']
    df = df.append(df_tmp)
    # df['row_id'] = test['row_id']
    # df_result = df_result.append(df_tmp['row_id'])

    print(t, mall, cur_index, len(test_all[test_all['mall_id'] == mall]))
df.to_csv('../output/stack/row_id.csv', index=False)
# df_result.to_csv('../output/stack_xgb_lgb_ovr_rf_et{}.csv'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), index=False)
print(df_result.head())
print(len(df_result))
print(len(df_result['row_id'].dropna()))
