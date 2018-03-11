import pandas as pd
import collections

def wifi_feat(df):
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


def wifi(wifi):
    wifi = sorted([wifi.split('|') for wifi in line[6].split(';')], key=lambda x: int(x[1]), reverse=True)
    return wifi

df_tr = pd.read_csv('../../data/train/m_615.csv')
df = pd.DataFrame()
for idx, row in df_tr.iterrows():
    dict_tmp = {}
    wifi = sorted([wifi.split('|') for wifi in row['wifi_infos'].split(';')], key=lambda x: int(x[1]), reverse=True)
    for i, each_wifi in enumerate(wifi, start=1):
        dict_tmp['bssid_{}'.format(i)] = each_wifi[0]
        dict_tmp['signal_{}'.format(i)] = each_wifi[1]
        dict_tmp['wifi_flag_{}'.format(i)] = each_wifi[2]

    df_tmp = pd.DataFrame.from_dict(data=dict_tmp, orient='index')
    df = df.append(df_tmp.T)
wifi_flag = {'false': 0, 'true': 1}
wifi_cols = [x for x in df.columns if 'wifi_flag' in x]
df[wifi_cols] = df[wifi_cols].map(wifi_flag)

print(df.head())

