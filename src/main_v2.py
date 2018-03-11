import os
import gc
from multiprocessing import Pool, cpu_count
import pickle

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from datetime import datetime
from feat_engineering import *
from models import *
from warnings import filterwarnings

filterwarnings('ignore')

path = '../data_ori/'

def get_iter():
    from csv import DictReader
    iter_dict = {}
    fi = open('../data/result_89.9099.csv', 'r')
    for row in DictReader(fi):
        v = int(row['best_iteration'])
        v = (int(v / 50) + 1) * 50
        iter_dict[row['mall_id']] = v

    print(iter_dict)
    return iter_dict

if __name__ == '__main__':
    start = datetime.now()
    iter_dict = get_iter()
    path = '../data_ori/'
    shop_info = pd.read_csv(path + 'shop_info.csv', usecols=['shop_id', 'mall_id'])
    train_all = pd.read_csv(path + 'train.csv', usecols=['shop_id', 'longitude', 'latitude', 'wifi_infos'])
    train_all = train_all.drop_duplicates()
    train_all = pd.merge(left=train_all, right=shop_info, how='left', on='shop_id')
    test_all = pd.read_csv(path + 'test.csv', usecols=['row_id', 'mall_id', 'longitude', 'latitude', 'wifi_infos'])

    df_result = pd.DataFrame()
    df_lgb_result = pd.DataFrame()
    df_xgb_result = pd.DataFrame()
    df_nn_result = pd.DataFrame()
    df_ovr_result = pd.DataFrame()
    df_rf_result = pd.DataFrame()
    df_et_result = pd.DataFrame()
    df_three_result = pd.DataFrame()

    for t, mall in enumerate(train_all['mall_id'].unique(), start=1):
        train = train_all[train_all['mall_id'] == mall].reset_index(drop=True)
        test = test_all[test_all['mall_id'] == mall].reset_index(drop=True)

        train_all = train_all[train_all['mall_id'] != mall]

        label = train['shop_id']
        test_row_id = test['row_id']

        print(t, mall, 'starting...')
        cache = False
        if cache:
            wt_train = open('../data/train/{}.pkl'.format(mall), 'rb')
            wt_test = open('../data/test/{}.pkl'.format(mall), 'rb')
            train = pickle.load(wt_train)
            test = pickle.load(wt_test)

        else:
            train_wifi, test_wifi = wifi_encode(train, test)  # 将bssid当作特征, signal作为值

            train = train[['longitude', 'latitude']]
            test = test[['longitude', 'latitude']]
            train = pd.concat([train, train_wifi], axis=1)
            test = pd.concat([test, test_wifi], axis=1)

            wt_train = open('../data/train/{}.pkl'.format(mall), 'wb')
            wt_test = open('../data/test/{}.pkl'.format(mall), 'wb')
            pickle.dump(train, wt_train)
            pickle.dump(test, wt_test)

            train_wifi = []
            train_feat = []

        # train, test = feat_encode(train, test)

        # 训练并预测
        y_train, num_class, lbl = label_encode(label)
        ovr_pred = ovr_prob(train, y_train, test)
        rf_pred = rf_prob(train, y_train, test)
        et_pred = et_prob(train, y_train, test)
        lgb_pred = lgb_prob(train, y_train, test, num_class, iter_dict[mall])
        xgb_pred = xgb_prob(train, y_train, test, num_class)

        train_nn, test_nn = select_feature(train, label, test)  # 特征选择
        train_nn, test_nn = feat_encode(train, test)
        nn_pred = nn_prob(train_nn, y_train, test_nn, num_class)


        df_lgb_result = df_lgb_result.append(pd.DataFrame(lgb_pred))
        df_xgb_result = df_xgb_result.append(pd.DataFrame(xgb_pred))
        df_nn_result = df_nn_result.append(pd.DataFrame(nn_pred))
        df_ovr_result = df_ovr_result.append(pd.DataFrame(ovr_pred))
        rf_pred = rf_prob(train, y_train, test)
        df_rf_result = df_rf_result.append(pd.DataFrame(rf_pred))
        df_et_result = df_et_result.append(pd.DataFrame(et_pred))

        pred = 0.35*np.array(lgb_pred) + 0.4*np.array(xgb_pred) + 0.25*np.array(ovr_pred)
        predict = []
        for x in pred:
            x = list(x)
            predict.append(x.index(sorted(x, reverse=True)[0]))
        predict = pd.DataFrame(predict, columns=['shop_id'])
        predict['shop_id'] = predict['shop_id'].apply(lambda x: lbl.inverse_transform(int(x)))

        test = pd.DataFrame(test_row_id)
        test['shop_id'] = predict['shop_id']
        result = test[['row_id', 'shop_id']]
        result['row_id'] = result['row_id'].astype('int')
        result['shop_id'] = result['shop_id']
        df_result = df_result.append(result)

        pred = 0.3*np.array(lgb_pred) + 0.3*np.array(xgb_pred) + 0.1*np.array(nn_pred) + 0.3*np.array(ovr_pred)
        predict = []
        for x in pred:
            x = list(x)
            predict.append(x.index(sorted(x, reverse=True)[0]))
        predict = pd.DataFrame(predict, columns=['shop_id'])
        predict['shop_id'] = predict['shop_id'].apply(lambda x: lbl.inverse_transform(int(x)))
        test = pd.DataFrame(test_row_id)
        test['shop_id'] = predict['shop_id']
        result = test[['row_id', 'shop_id']]
        result['row_id'] = result['row_id'].astype('int')
        result['shop_id'] = result['shop_id']
        df_three_result = df_three_result.append(result)

        train = []
        test = []
        train_nn = []
        test_nn = []
        predict = []
        pre_result = []
        result = []
        del train, test, predict, pre_result, result, train_nn, test_nn
        gc.collect()

        print(t, mall, datetime.now() - start)

    df_lgb_result.to_csv('../output/lgb_prob.csv', index=False)
    df_xgb_result.to_csv('../output/xgb_prob.csv', index=False)
    df_rf_result.to_csv('../output/rf_prob.csv', index=False)
    df_et_result.to_csv('../output/et_prob.csv', index=False)
    df_nn_result.to_csv('../output/nn_prob.csv', index=False)
    df_ovr_result.to_csv('../output/ovr_prob.csv', index=False)
    df_result.to_csv('../output/stack_lgb_xgb_ovr{}.csv'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), index=False)
    df_three_result.to_csv('../output/stack_four.csv', index=False)

print(datetime.now())
