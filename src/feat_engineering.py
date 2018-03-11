import pandas as pd
import gc
import numpy as np
from multiprocessing import Pool, cpu_count

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


def parallel(df, func):
    if len(df) > 0:
        # print(df.shape)
        p = Pool(cpu_count())
        df = p.map(func, np.array_split(df, cpu_count()))
        df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
        p.close();
        p.join()
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
        wifi = sorted([wifi.split('|') for wifi in row['wifi_infos'].split(';')], key=lambda x: int(x[1]),
                      reverse=True)[:5]
        for i, each_wifi in enumerate(wifi, start=1):
            dict_tmp['bssid_{}'.format(i)] = int(each_wifi[0][2:])
            dict_tmp['signal_{}'.format(i)] = int(each_wifi[1])
            # dict_tmp['wifi_flag_{}'.format(i)] = each_wifi[2]
        df_tmp = pd.DataFrame.from_dict(data=dict_tmp, orient='index')
        df_feat = df_feat.append(df_tmp.T)
        del dict_tmp, df_tmp
    # wifi_flag = {'false': 0, 'true': 1}
    # wifi_cols = [x for x in df_feat.columns if 'wifi_flag' in x]
    # for col in wifi_cols:
    #     df_feat[col] = df_feat[col].map(wifi_flag)

    del df
    gc.collect()
    return df_feat


#
def wifi_filter(train, test):
    wifi_dict = {}
    ll = []
    wifi_train = []  # train中出现的wifi
    wifi_test_only = []  # 仅出现在test中，而train中没有的wifi
    for index, row in train.iterrows():
        wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
        for wifi in wifi_list:
            wifi_train.append(wifi[0])
            if wifi[0] not in wifi_dict:
                wifi_dict[wifi[0]] = 1
            else:
                wifi_dict[wifi[0]] += 1

    for index, row in test.iterrows():
        wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
        for wifi in wifi_list:
            if wifi[0] not in wifi_dict:
                wifi_dict[wifi[0]] = 1
            else:
                wifi_dict[wifi[0]] += 1
            if wifi[0] not in wifi_train:
                wifi_test_only.append(wifi[0])
            else:
                ll.append(wifi[0])

    delete_wifi = []
    for wifi in wifi_dict:
        if wifi_dict[wifi] < 5:  # 总的出现次数少于20次的wifi，删掉
            delete_wifi.append(wifi)

    df_tr = pd.DataFrame()
    for index, row in train.iterrows():
        wifi_list = [wifi for wifi in row['wifi_infos'].split(';')]
        wifis = ''
        for wifi in wifi_list:
            if wifi.split('|')[0] not in delete_wifi:
                wifis += wifi + ';'

        df_tr = df_tr.append(pd.DataFrame([wifis]))
    df_tr.columns = ['wifi_infos']
    train['wifi_infos'] = df_tr['wifi_infos'].reset_index(drop=True)
    df_te = pd.DataFrame()

    for index, row in test.iterrows():
        wifi_list = [wifi for wifi in row['wifi_infos'].split(';')]
        wifis = ''
        for wifi in wifi_list:
            if wifi.split('|')[0] not in delete_wifi and wifi.split('|')[0] in wifi_train:
                wifis += wifi + ';'

        df_te = df_te.append(pd.DataFrame([wifis]))
    df_te.columns = ['wifi_infos']
    test['wifi_infos'] = df_te['wifi_infos'].reset_index(drop=True)

    return train, test


# 将bssid当作特征, signal作为值
def wifi_encode(train, test):
    df_all = pd.concat([train, test], axis=0).reset_index(drop=True)
    l = []
    wifi_dict = {}
    for index, row in df_all.iterrows():
        r = {}
        wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
        for i in wifi_list:
            r[i[0]] = int(i[1])
            if i[0] not in wifi_dict:
                wifi_dict[i[0]] = 1
            else:
                wifi_dict[i[0]] += 1
        l.append(r)

    wifi_train = []
    for index, row in train.iterrows():
        wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
        for i in wifi_list:
            wifi_train.append(i[0])

    delete_wifi = []
    for i in wifi_dict:
        if wifi_dict[i] < 20:
            delete_wifi.append(i)
    m = []
    for row in l:
        new = {}
        for n in row.keys():
            if n not in delete_wifi and n in wifi_train:
                new[n] = row[n]
        m.append(new)

    df_all = pd.concat([df_all, pd.DataFrame(m)], axis=1)
    df_train = df_all[df_all.shop_id.notnull()].reset_index(drop=True)
    df_test = df_all[df_all.shop_id.isnull()].reset_index(drop=True)
    df_all = []
    feature = [x for x in df_train.columns if 'b_' in x]
    df_train = df_train[feature]
    df_test = df_test[feature]
    return df_train, df_test


def feat_encode(df_tr, df_te):
    df_all = pd.concat([df_tr, df_te], axis=0)
    df_all = df_all.fillna(-999).replace(np.inf, -999)
    # cat_eats = [x for x in df_all.columns]
    # for f in cat_eats:
    #     lbl = LabelEncoder()
    #     lbl.fit(list(df_all[f].values))
    #     df_all[f] = lbl.transform(list(df_all[f].values))

    x_train = df_all.iloc[:df_tr.shape[0], :]
    x_test = df_all.iloc[df_tr.shape[0]:, :]
    df_all = []

    scale = StandardScaler()
    scale.fit(x_train)
    x_train = scale.transform(x_train)
    x_test = scale.transform(x_test)

    return x_train, x_test


def wifi_tfidf(train, test):
    train_num = train.shape[0]
    df_all = pd.concat([train, test], axis=0)
    tfidf = CountVectorizer(stop_words='english', max_features=300)
    all_sparse = tfidf.fit_transform(df_all["wifi_infos"].values.astype('U')).toarray()
    tr_sparse = pd.DataFrame(all_sparse[:train_num])
    te_sparse = pd.DataFrame(all_sparse[train_num:])
    return tr_sparse, te_sparse


def label_encode(df_label):
    lbl = LabelEncoder()
    lbl.fit(list(df_label.values))
    df_label = lbl.transform(list(df_label.values))
    num_class = df_label.max() + 1
    return df_label, num_class, lbl


def time_feat(df):
    from datetime import datetime
    col = 'time_stamp'
    # df['{}_year'.format(col)] = df[col].apply(lambda x: int(str(x)[:4]) if len(str(x)) > 3 else np.NaN)
    df['{}_month'.format(col)] = df[col].apply(lambda x: int(str(x)[5:7]) if len(str(x)) > 3 else np.NaN)
    df['{}_date'.format(col)] = df[col].apply(lambda x: int(str(x)[8:10]) if len(str(x)) > 3 else np.NaN)
    df['{}_hour'.format(col)] = df[col].apply(lambda x: int(str(x)[11:13]) if len(str(x)) > 3 else np.NaN)
    df['{}_wkday'.format(col)] = df[col].apply(lambda x: int(datetime(int(str(x)[:4]) if len(str(x)) > 3 else np.NaN,
                                                                  int(str(x)[5:7]) if len(str(x)) > 3 else np.NaN,
                                                                  int(str(x)[8:10]) if len(str(x)) > 3 else np.NaN).strftime('%w')))
    return df


from sklearn.feature_selection import SelectFromModel


# def select_feature(x_train, y_train, x_test):  # 太耗时
#     x_train = x_train.fillna(-999)
#     x_test = x_test.fillna(-999)
#     clf = GradientBoostingClassifier()
#     clf.fit(x_train, y_train)
#     model = SelectFromModel(clf, prefit=True, threshold="mean")
#     x_train = pd.DataFrame(model.transform(x_train))
#     x_test = pd.DataFrame(model.transform(x_test))
#     return x_train, x_test


def select_feature(x_train, y_train, x_test):
    y_train, num_class, lbl = label_encode(y_train)
    clf = XGBClassifier(
        learning_rate=0.1,  # 默认0.3
        n_estimators=20,  # 树的个数
        max_depth=5,
        min_child_weight=1,
        gamma=0.5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',  # 逻辑回归损失函数
        nthread=8,  # cpu线程数
        scale_pos_weight=1,
        reg_alpha=1e-05,
        reg_lambda=1,
        seed=2017)  # 随机种子

    clf.fit(x_train, y_train)
    new_feature = clf.apply(x_train)
    new_feature = pd.DataFrame(new_feature)
    x_train = pd.concat([x_train, new_feature], axis=1)
    # x_train = new_feature
    new_feature = clf.apply(x_test)
    new_feature = pd.DataFrame(new_feature)
    x_test = pd.concat([x_test, new_feature], axis=1)
    # new_feature = []
    # x_test = new_feature
    return x_train, x_test
