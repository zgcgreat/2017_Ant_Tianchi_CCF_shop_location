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

if __name__ == '__main__':
    start = datetime.now()

    path = '../data_ori/'
    shop_info = pd.read_csv(path + 'shop_info.csv', usecols=['shop_id', 'mall_id'])
    train_all = pd.read_csv(path + 'train.csv', usecols=['shop_id', 'time_stamp', 'longitude', 'latitude', 'wifi_infos'])
    train_all = train_all.drop_duplicates()
    train_all = pd.merge(left=train_all, right=shop_info, how='left', on='shop_id')
    test_all = pd.read_csv(path + 'test.csv', usecols=['row_id', 'mall_id', 'time_stamp', 'longitude', 'latitude', 'wifi_infos'])

    df_result = pd.DataFrame()
    for t, mall in enumerate(train_all['mall_id'].unique(), start=1):
        train = train_all[train_all['mall_id'] == mall].reset_index(drop=True)
        test = test_all[test_all['mall_id'] == mall].reset_index(drop=True)

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

            # train, test = wifi_filter(train, test)

            # train_feat = parallel(train, wifi_feat)　# 基本没效果
            # test_feat = parallel(test, wifi_feat)

            # tr_wifi_tfidf, te_wifi_tfidf = wifi_tfidf(pd.DataFrame(train['wifi_infos']),  # 效果变差
            #                                           pd.DataFrame(test['wifi_infos']))
            train = time_feat(train)
            test = time_feat(test)

            train = train.drop(['shop_id', 'time_stamp', 'wifi_infos', 'mall_id'], axis=1)
            test = test.drop(['row_id', 'time_stamp', 'wifi_infos', 'mall_id'], axis=1)

            train = pd.concat([train, train_wifi], axis=1)
            test = pd.concat([test, test_wifi], axis=1)

            # train, test = select_feature(train, label, test)  # 特征选择
            # print(train.columns)

            # train, test = feat_encode(train, test)  # 对str类型特征做labelEncoder

            wt_train = open('../data/train/{}.pkl'.format(mall), 'wb')
            wt_test = open('../data/test/{}.pkl'.format(mall), 'wb')
            pickle.dump(train, wt_train)
            pickle.dump(test, wt_test)

            train_wifi = []
            train_feat = []

        train, test = feat_encode(train, test)

        # 训练并预测
        predict, train_acc = rf(train, label, test)

        test = pd.DataFrame(test_row_id)
        test['shop_id'] = predict['shop_id']
        result = test[['row_id', 'shop_id']]
        result['row_id'] = result['row_id'].astype('int')
        result['shop_id'] = result['shop_id']
        df_result = df_result.append(result)

        train_all = train_all[train_all['mall_id'] != mall]

        train = []
        test = []
        predict = []
        pre_result = []
        result = []
        del train, test, predict, pre_result, result
        gc.collect()
        print(t, mall, train_acc, datetime.now() - start, datetime.now())

    df_result.to_csv('../output/rf_result{}.csv'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), index=False)

print(datetime.now())
