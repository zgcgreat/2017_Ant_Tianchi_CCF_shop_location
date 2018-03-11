from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np


def parallel(df, func):
    if len(df) > 0:
        # print(df.shape)
        p = Pool(cpu_count())
        df = p.map(func, np.array_split(df, cpu_count()))
        df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
        p.close();
        p.join()
        return df

import pandas as pd
import os

path = '../output/stack/'


def vote(df):
    res = pd.DataFrame()
    for id in df['row_id'].unique():
        df1 = df[df['row_id'] == id]
        tmp = df1['shop_id'].value_counts().reset_index()
        tmp.columns = ['shop_id', 'count']
        tmp.insert(0, 'row_id', id)
        res = res.append(tmp)

    res = res.sort_values(by=['count'], ascending=False)
    res = res.drop_duplicates(subset='row_id', keep='first')
    res = res.drop(['count'], axis=1)
    return res


if __name__ == '__main__':
    files = os.listdir(path)

    df = pd.DataFrame()
    for f in files:
        df_tmp = pd.read_csv(path + f)
        df = df.append(df_tmp)
    df = df.sort_values(by='row_id', ascending=False)
    print(df.head())
    res = parallel(df, vote)
    res = res.drop_duplicates()
    print(res.head())
    print(len(res))
    res.to_csv('../output/stack/stack_result{}.csv'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), index=False)
