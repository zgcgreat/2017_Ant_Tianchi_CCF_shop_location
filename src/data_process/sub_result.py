import os
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import collections
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime

path = '../../output/result/'


def parallel(df, func):
    if len(df) > 0:
        # print(df.shape)
        p = Pool(cpu_count())
        df = p.map(func, np.array_split(df, cpu_count()))
        df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
        p.close(); p.join()
        return df


def vote(path):
    files = os.listdir(path)
    res = pd.DataFrame()
    for t, f in enumerate(files, start=1):
        print(t, f)
        df = pd.read_csv(path + f)
        for id in df['row_id'].unique():
            df1 = df[df['row_id'] == id]
            tmp = df1['shop_id'].value_counts().reset_index()
            tmp.columns = ['shop_id', 'count']
            tmp.insert(0, 'row_id', id)
            # tmp = tmp.sort_values(by=['count'], ascending=False)
            # tmp = tmp.drop_duplicates(subset='row_id', keep='first')
            # tmp = tmp.drop(['count'], axis=1)
            res = res.append(tmp)
    res = res.sort_values(by=['count'], ascending=False)
    res = res.drop_duplicates(subset='row_id', keep='first')
    res = res.drop(['count'], axis=1)
    res['shop_id'] = res['shop_id'].apply(lambda x: 's_' + str(x))
    return res


def sub_result(path):
    files = os.listdir(path)
    res = pd.DataFrame()
    for t, f in enumerate(files, start=1):
        print(t, f)
        df = pd.read_csv(path + f)
        res = res.append(df)
    res['shop_id'] = res['shop_id'].apply(lambda x: 's_' + str(x))
    return res


if __name__ == '__main__':
    start = datetime.now()
    # res = vote(path)
    res = sub_result(path)
    test = pd.read_csv('../../data_ori/test.csv', usecols=['row_id'])
    print(len(test), len(res))
    res = pd.merge(left=test, right=res, how='left', on='row_id')
    print(res.head())
    res.to_csv('../../output/ovr_result{}.csv'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), index=False)
    print(datetime.now() - start)
