import os
import gc
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from datetime import datetime

path = '../../data_ori/'

'''
AB榜测试集-evaluation_public.csv  483931
训练数据-ccf_first_round_shop_info.csv  shop_id 8477  mall_id 97
训练数据-ccf_first_round_user_shop_behavior.csv  1138015
'''


def parallel(df, func):
    if len(df) > 0:
        # print(df.shape)
        p = Pool(cpu_count())
        df = p.map(func, np.array_split(df, cpu_count()))
        df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
        p.close(); p.join()
        return df



def wifi_feature(df):
    s = df['wifi_infos'].str.split(';').apply(pd.Series, 1).stack()
    s.index = s.index.droplevel(-1)
    s.name = 'wifi_info'
    df = df.join(s)

    df = df.reset_index()
    df['bssid'] = df.wifi_info.apply(lambda s: s.split('|')[0])
    df['signal'] = df.wifi_info.apply(lambda s: s.split('|')[1])
    df['wifi_flag'] = df.wifi_info.apply(lambda s: s.split('|')[2])
    df.drop(['wifi_infos', 'wifi_info'], axis=1, inplace=True)

    wifi_flag = {'false': 0, 'true': 1}
    df['wifi_flag'] = df['wifi_flag'].map(wifi_flag)

    # 将bssid由string型转化为数值型，bssid样例：b_4162269
    df['bssid'] = df.bssid.apply(lambda s: s[2:])
    df['bssid'] = df.bssid.astype('int')

    # 将shop_id由string型转化为数值型，shop_id样例：s_1126
    if 'shop_id' in df.columns:
        df['shop_id'] = df.shop_id.apply(lambda s: s[2:])
        df['shop_id'] = df.shop_id.astype('int')
    return df


def wifi_feat(df):
    df_feat = pd.DataFrame()
    for t, row in df.iterrows():
        dict_tmp = {}
        if 'row_id' in row.keys():
            dict_tmp['row_id'] = int(row['row_id'])
        wifi = sorted([wifi.split('|') for wifi in row['wifi_infos'].split(';')], key=lambda x: int(x[1]), reverse=True)[:5]
        for i, each_wifi in enumerate(wifi, start=1):
            dict_tmp['bssid_{}'.format(i)] = int(each_wifi[0][2:])
            dict_tmp['signal_{}'.format(i)] = int(each_wifi[1])
            dict_tmp['wifi_flag_{}'.format(i)] = each_wifi[2]
        df_tmp = pd.DataFrame.from_dict(data=dict_tmp, orient='index')
        df_feat = df_feat.append(df_tmp.T)
        del dict_tmp, df_tmp
    wifi_flag = {'false': 0, 'true': 1}
    wifi_cols = [x for x in df_feat.columns if 'wifi_flag' in x]
    for col in wifi_cols:
        df_feat[col] = df_feat[col].map(wifi_flag)
    # 将shop_id由string型转化为数值型，shop_id样例：s_1126
    # if 'shop_id' in df.columns:
    #     df['shop_id'] = df.shop_id.apply(lambda s: s[2:])
    #     df['shop_id'] = df.shop_id.astype('int')
    # df_feat['shop_id'] = df['shop_id'].astype('int')
    # df_feat['signal_sum'] = sum(df_feat['signal_1'] + df_feat['signal_2'] + df_feat['signal_3'])
    del df
    gc.collect()
    return df_feat




def grid_knn(train, label, test):
    knn = KNeighborsClassifier(algorithm='kd_tree')
    k_range = list(range(1, 20))
    param_gridknn = dict(n_neighbors=k_range)
    gridKNN = GridSearchCV(knn, param_gridknn, cv=3, scoring='accuracy', verbose=1, error_score=0, n_jobs=-1)
    gridKNN.fit(train, label)
    predict = gridKNN.predict(test.drop(['row_id'], axis=1))
    predict = pd.DataFrame(predict, columns=['shop_id'])
    print('acc:', accuracy_score(label, gridKNN.predict(train)))
    print(gridKNN.predict(train))
    return predict, gridKNN.best_params_


def knn(train, label, test):
    knn = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', metric='manhattan', metric_params=None, n_jobs=-1)
    knn.fit(train, label)
    predict = knn.predict(test.drop(['row_id'], axis=1))
    predict = pd.DataFrame(predict, columns=['shop_id'])
    print('acc:', accuracy_score(label, knn.predict(train)))
    return predict


def rf(train, label, test):
    randomforest = RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1, random_state=2017,
                                          max_features="auto", verbose=0)
    randomforest.fit(train, label)
    predict = randomforest.predict(test.drop(['row_id'], axis=1))
    predict = pd.DataFrame(predict, columns=['shop_id'])
    print('acc:', accuracy_score(label, randomforest.predict(train)))
    return predict


def ovr(train, label, test):
    if len(train) > 20000:
        max_depth = 10
        n_estimators = 100
    else:
        max_depth = 6
        n_estimators = 50
    print(len(train), len(test))
    # x_train, y_train, x_val, y_val = train_test_split(train, label, test_size=0.1, random_state=2017)
    est = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=2017, max_features="auto",
                                 verbose=0)
    ovr = OneVsRestClassifier(est, n_jobs=-1)
    ovr.fit(train, label)
    pred = ovr.predict(test.drop(['row_id'], axis=1))
    pred = pd.DataFrame(pred, columns=['shop_id'])
    print('train_acc:', accuracy_score(label, ovr.predict(train)))
    return pred


if __name__ == '__main__':
    start = datetime.now()
    malls = os.listdir('../../data/train/')

    for t, mall in enumerate(malls, start=1):
        print(t, mall)
        train = pd.read_csv('../../data/train/'+mall, usecols=['shop_id', 'wifi_infos'])
        test = pd.read_csv('../../data/test/' + mall, usecols=['row_id', 'wifi_infos'])


        #只用wifi_infos
        # train = wifi_feature(train)
        label = train['shop_id'].apply(lambda x: int(x[2:]))
        train = train.drop(['shop_id'], axis=1)
        train = parallel(train, wifi_feat)

        test = parallel(test, wifi_feat)

        train = train.fillna(-999)
        test = test.fillna(-999)

        predict = ovr(train, label, test)

        # predict = rf(train, label, test)
        pre_result = pd.concat([test['row_id'], predict['shop_id']], axis=1)
        result = pd.DataFrame(columns=['row_id', 'shop_id'])
        result = pd.concat([result, pre_result], axis=0)
        result['row_id'] = result['row_id'] .astype('int')
        result['shop_id'] = result['shop_id'].astype('int')

        result.to_csv('../../output/result/result_' + mall, index=None)

        train = []
        test = []
        predict = []
        pre_result = []
        result = []
        del train, test, predict, pre_result, result
        gc.collect()
        print(datetime.now() - start)

