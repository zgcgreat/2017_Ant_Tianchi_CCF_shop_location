import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid
import numpy as np

shop_path = '../data/shop_info.csv'
eval_info_path = 'result_89.9099.csv'
shop_info = pd.read_csv(shop_path)
mall_list = list(set(shop_info['mall_id'].values))
eval_info = pd.read_csv(eval_info_path)
for i, m in enumerate(mall_list):
    save_path = 'multi_data/' + m
    train_feat = pd.read_csv(save_path + '/train_feat.csv')
    train_label = train_feat.pop('label').values
    feat_names = list(train_feat.columns)
    categorical_feat_names = ['wday']

    test_feat = pd.read_csv(save_path + '/test_feat.csv')
    label_mapping = pd.read_csv(save_path + '/label_mapping.csv')
    labels = label_mapping['label'].values
    shops = label_mapping['shop_id'].values
    map_dict = {}
    for j in range(len(labels)):
        map_dict[labels[j]] = shops[j]

    print('mall_id:' + m + ' (' + str(i + 1) + '/' + str(len(mall_list)) + ')')
    best_iterations = eval_info[(eval_info['mall_id'] == m)]['best_iteration'].values[0]
    best_iterations = (int(best_iterations / 50) + 1) * 50

    params = {
        'num_class': [max(labels) + 1],
        'metric': ['multi_error'],
        'objective': ['multiclass'],
        'learning_rate': [0.15],
        'feature_fraction': [0.8],
        'max_depth': [13],
        'num_leaves': [200],
        'bagging_fraction': [0.8],
        'bagging_freq': [5],
        'min_data_in_leaf': [15],
        'min_gain_to_split': [0],
        'num_iterations': [best_iterations],
        'lambda_l1': [0.01],
        'lambda_l2': [1],
        'verbose': [0],
        'is_unbalance': [True]
    }
    params = list(ParameterGrid(params))
    lgbtrain = lgb.Dataset(train_feat, label=train_label, feature_name=feat_names,
                           categorical_feature=categorical_feat_names)
    lgbtest = test_feat[feat_names]
    for param in params:
        clf = lgb.train(param, lgbtrain, num_boost_round=param['num_iterations'],
                        categorical_feature=categorical_feat_names)
        pred = clf.predict(lgbtest)
        predict_label = np.argmax(pred, axis=1)
        rows = test_feat['row_id'].values
        shop_ids = []
        for l in predict_label:
            shop_ids.append(map_dict[l])
        results = pd.DataFrame([list(rows), list(shop_ids)], index=['row_id', 'shop_ids'])
        results = results.transpose()
        results.to_csv(save_path + '/result.csv', index=False)
