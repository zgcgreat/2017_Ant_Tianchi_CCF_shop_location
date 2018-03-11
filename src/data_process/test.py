import os
from datetime import datetime
from multiprocessing import Pool, cpu_count

import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '../../output/'


def parallel(df, func):
    if len(df) > 0:
        # print(df.shape)
        p = Pool(cpu_count())
        df = p.map(func, np.array_split(df, cpu_count()))
        df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
        p.close();
        p.join()
        return df


def wifi_feat(df):
    df_feat = pd.DataFrame()
    for t, row in df.iterrows():
        dict_tmp = {}
        if 'row_id' in row.keys():
            dict_tmp['row_id'] = int(row['row_id'])
        wifi = sorted([wifi.split('|') for wifi in row['wifi_infos'].split(';')], key=lambda x: int(x[1]),
                      reverse=True)[:10]
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

    del df
    gc.collect()
    return df_feat


if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    df_rule = pd.read_csv(path+'rule_result.csv')
    df_rule = df_rule.rename(columns={'row_id': 'row_id_rule', 'shop_id': 'shop_id_rule'})

    df_ovr = pd.read_csv(path+'ovr_result20171030-090837.csv')

    df = pd.concat([df_rule, df_ovr], axis=1)

    len_all = len(df)

    df = df[df['shop_id_rule'] == df['shop_id']]
    df = df[['row_id', 'shop_id']]

    print(len(df) / len_all)

    test = pd.read_csv('../../data_ori/test.csv')  # , usecols=['row_id', 'mall_id', 'wifi_infos']

    df = pd.merge(left=test, right=df, how='left', on='row_id')
    print(len(df))
    df = df.fillna(0)
    df = df[df['shop_id'] == 0]
    # df = df[df['mall_id'] == 'm_5085']
    # print(df.head())
    # print(len(df))

    print(len(df['mall_id'].unique()))

    # df.to_csv(path + 'rule_ovr_result.csv', index=False)

    df_tmp = df['mall_id'].value_counts()
    print(df_tmp)

    # test = test[test['mall_id'] == 'm_5085']
    # test = parallel(test,wifi_feat)
    #
    # print(test.head())