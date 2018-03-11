import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid
import numpy as np
import h5py
import matplotlib.pyplot as plt

train_feat = pd.read_csv('train_feat.csv')
train_label = train_feat.pop('label').values

validation_feat = pd.read_csv('validation_feat.csv')
validation_label = validation_feat.pop('label').values
feat_names = list(train_feat.columns)
categorical_feat_names = ['wday']

label_mapping = pd.read_csv('multi_data/m_7800' + '/label_mapping.csv')
labels = label_mapping['label'].values
shops = label_mapping['shop_id'].values

params = {
    'num_class': [max(labels) + 1],
    'metric': ['multi_error'],
    'objective': ['multiclass'],
    'learning_rate': [0.15],
    'feature_fraction': [0.6],
    'max_depth': [13],
    'num_leaves': [200],
    'bagging_fraction': [0.8],
    'bagging_freq': [5],
    'min_data_in_leaf': [15],
    'min_gain_to_split': [0],
    'num_iterations': [500],
    'lambda_l1': [0.01],
    'lambda_l2': [1],
    'verbose': [0],
    'is_unbalance': [True]
}
params = list(ParameterGrid(params))
lgbtrain = lgb.Dataset(train_feat, label=train_label, feature_name=feat_names,
                       categorical_feature=categorical_feat_names)  #
lgbeval = lgb.Dataset(validation_feat, label=validation_label, reference=lgbtrain, feature_name=feat_names,
                      categorical_feature=categorical_feat_names)  #
lgbtest = validation_feat
for param in params:
    print(param)
    clf = lgb.train(param, lgbtrain, valid_sets=lgbeval, num_boost_round=param['num_iterations'],
                    early_stopping_rounds=50, categorical_feature=categorical_feat_names)  #
    print('best interation:' + str(clf.best_iteration))
    pred = clf.predict(lgbtest)
    predict_label = np.argmax(pred, axis=1)
    result = validation_label - predict_label
    print('acc:' + str(len(np.nonzero(result == 0)[0]) / result.shape[0]))

# #输出错误案例
# wrong_index=np.nonzero(result!=0)[0]
# predict_label=predict_label[wrong_index]
# validation_label=validation_label[wrong_index]
# label_dict={}
# for i,l in enumerate(labels):
#     label_dict[l]=shops[i]
# predict_shop=[]
# true_shop=[]
# for l in predict_label:
#     predict_shop.append(label_dict[l])
# for l in validation_label:
#     true_shop.append(label_dict[l])
# #返回正确样本和错误样本的对应置信度
# predict_conf=[]
# true_conf=[]
# predict_rank=[]
# true_rank=[]
# for i in range(len(predict_shop)):
#     predict_conf.append(pred[wrong_index[i],predict_label[i]])
#     predict_rank.append(1)
#     true_conf.append(pred[wrong_index[i],validation_label[i]])
#     true_rank.append(pred.shape[1]-np.where(np.argsort(pred[wrong_index[i],:])==validation_label[i])[0][0])
# true_df=np.concatenate((np.reshape(np.array(true_shop),(-1,1)),np.reshape(np.array(true_conf),(-1,1)),np.reshape(np.array(true_rank),(-1,1))),axis=1)
# predict_df=pd.DataFrame(np.concatenate((np.reshape(np.array(predict_shop),(-1,1)),np.reshape(np.array(predict_conf),(-1,1)),np.reshape(np.array(predict_rank),(-1,1))),axis=1),columns=['shop_id','predict_conf','predict_rank'])
# shop_info=pd.read_csv('../data/shop_info.csv')
# predict_df=pd.merge(predict_df,shop_info,on='shop_id',how='left')
# true_df_ex=pd.read_csv('../data/train_data.csv')
# true_df_ex=pd.merge(true_df_ex,shop_info[['shop_id','mall_id']],on='shop_id',how='left')
# true_df_ex=true_df_ex[(true_df_ex['mall_id']=='m_7800')]
# true_df_ex=true_df_ex[(true_df_ex['time_stamp'] >= '2017-08-28 00:00:00')]
# true_df_ex=true_df_ex.iloc[wrong_index,:].values
# true_df=np.concatenate((true_df,true_df_ex),axis=1)
# all_df=pd.DataFrame(np.concatenate((true_df[:,[0,1,2,3,6,7,11]],predict_df.values[:,[0,1,2,4,5,9]]),axis=1),
# columns=['true_shop','true_conf','true_rank','user_id','true_lon','true_lat','true_wifi',
#          'predict_shop','predict_conf','predict_rank','predict_lon','predict_lat','predict_wifi'])
#
# all_df.to_csv('7800wrong.csv',index=False)
