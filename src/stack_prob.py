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
    label = train['shop_id']
    y_train, num_class, lbl = label_encode(label)

    cur_index = last_index + len(test_all[test_all['mall_id'] == mall])
    xgb = np.array(df_xgb)[last_index:cur_index]
    lgb = np.array(df_lgb)[last_index:cur_index]
    ovr = np.array(df_ovr)[last_index:cur_index]
    nn = np.array(df_nn)[last_index:cur_index]
    rf = np.array(df_rf)[last_index:cur_index]
    et = np.array(df_et)[last_index:cur_index]
    last_index = cur_index

    pred = 0.2*xgb + 0.2*lgb + 0.2*ovr + 0.2*rf + 0.15 * et + 0.05 * nn
    df = df.append(pd.DataFrame(pred))
    df['row_id'] = test['row_id']
    print(df['row_id'].head())
    # pd.DataFrame(pred).to_csv('../output/stack_all_prob{}.csv'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), index=False)
    # pred = xgb + lgb + ovr + rf + et
    predict = []
    for x in pred:
        x = list(x)
        predict.append(x.index(sorted(x, reverse=True)[0]))
    predict = pd.DataFrame(predict, columns=['shop_id'])
    predict['shop_id'] = predict['shop_id'].apply(lambda x: lbl.inverse_transform(int(x)))

    test_row_id = test['row_id']
    test = pd.DataFrame(test_row_id)
    test['shop_id'] = predict['shop_id']
    result = test[['row_id', 'shop_id']]
    result['row_id'] = result['row_id'].astype('int')
    result['shop_id'] = result['shop_id']
    df_result = df_result.append(result)

    print(t, mall, cur_index, len(test_all[test_all['mall_id'] == mall]))
# df.to_csv('../output/stack_all_prob{}.csv'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), index=False)
print(df_result.head())
# df_result.to_csv('../output/stack_xgb_lgb_ovr_rf_et{}.csv'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), index=False)
