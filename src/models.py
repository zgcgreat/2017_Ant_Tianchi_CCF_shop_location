import os
import gc
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from datetime import datetime
import xgboost
import lightgbm
from sklearn.svm import LinearSVC

from feat_engineering import *
import warnings
warnings.filterwarnings('ignore')


def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'accuracy', accuracy_score(labels, preds)


def lgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'accuracy', accuracy_score(labels, preds)


def lr(train, label, test):
    logisticregression = LogisticRegression(n_jobs=-1, random_state=2017, C=0.1, max_iter=2000)
    logisticregression.fit(train, label)
    predict = logisticregression.predict(test)
    predict = pd.DataFrame(predict, columns=['shop_id'])
    train_acc = accuracy_score(label, logisticregression.predict(train))
    return predict, train_acc


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
    pca = PCA(n_components=500)
    pca.fit(train)
    train = pca.transform(train)
    test = pca.transform(test)
    knn = KNeighborsClassifier(n_neighbors=5, algorithm='auto', metric='manhattan', metric_params=None, n_jobs=-1)
    knn.fit(train, label)
    predict = knn.predict(test)
    predict = pd.DataFrame(predict, columns=['shop_id'])
    train_acc = accuracy_score(label, knn.predict(train))
    return predict, train_acc


def ovr_knn(train, label, test):
    train = train.fillna(-999)
    test = test.fillna(-999)
    est = KNeighborsClassifier(n_neighbors=5, algorithm='auto', p=1, metric_params=None, n_jobs=-1)
    ovr = OneVsRestClassifier(est, n_jobs=-1)
    ovr.fit(train, label)
    pred = ovr.predict(test)
    pred = pd.DataFrame(pred, columns=['shop_id'])
    train_acc = accuracy_score(label, ovr.predict(train))
    return pred, train_acc


def rf(train, label, test):
    # train = train.fillna(-999)
    # test = test.fillna(-999)
    randomforest = RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=2017,
                                          max_features="auto", verbose=0)
    randomforest.fit(train, label)
    predict = randomforest.predict(test)
    predict = pd.DataFrame(predict, columns=['shop_id'])
    train_acc = accuracy_score(label, randomforest.predict(train))
    return predict, train_acc


def ada(train, label, test):
    adaboost = AdaBoostClassifier(n_estimators=1000, random_state=2017, learning_rate=0.01)
    adaboost.fit(train, label)
    predict = adaboost.predict(test)
    predict = pd.DataFrame(predict, columns=['shop_id'])
    train_acc = accuracy_score(label, adaboost.predict(train))
    return predict, train_acc


def et(train, label, test):
    extratree = ExtraTreesClassifier(n_estimators=300, max_depth=None, max_features="auto", n_jobs=-1, random_state=2017,
                                     verbose=0)
    extratree.fit(train, label)
    predict = extratree.predict(test)
    predict = pd.DataFrame(predict, columns=['shop_id'])
    train_acc = accuracy_score(label, extratree.predict(train))
    extratree = []
    del extratree
    return predict, train_acc


def ovr(train, label, test):
    train = train.fillna(-999)
    test = test.fillna(-999)
    n_estimators = 400
    max_depth = None
    est = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=2017,
                                 max_features="auto",
                                 verbose=0)
    ovr = OneVsRestClassifier(est, n_jobs=-1)
    ovr.fit(train, label)
    pred = ovr.predict(test)
    pred = pd.DataFrame(pred, columns=['shop_id'])
    train_acc = accuracy_score(label, ovr.predict(train))
    return pred, train_acc


def lsvc(train, label, test):
    linearsvc = LinearSVC(random_state=2017)
    linearsvc.fit(train, label)
    predict = linearsvc.predict(test)
    predict = pd.DataFrame(predict, columns=['shop_id'])
    train_acc = accuracy_score(label, linearsvc.predict(train))
    return predict, train_acc


def fm(train, label, test):
    from fastFM import als
    FM = als.FMClassification(n_iter=1000, init_stdev=0.1, rank=8, l2_reg_w=0.2, l2_reg_V=0.5, )
    FM.fit(train, label)
    predict = FM.predict(test)
    train_acc = accuracy_score(label, FM.predict(train))
    return predict, train_acc


def xgb(x_train, y_train, x_test):
    y_train, num_class, lbl = label_encode(y_train)

    params = {
        'objective': 'multi:softmax',
        'eta': 0.1,
        'max_depth': 5,
        'eval_metric': 'merror',
        'seed': 0,
        'missing': -999,
        'num_class': num_class,
        'silent': 1
    }
    xgb_train = xgboost.DMatrix(x_train, label=y_train)
    xgb_test = xgboost.DMatrix(x_test)
    watchlist = [(xgb_train, 'train'), (xgb_train, 'test')]
    num_rounds = 500
    model = xgboost.train(params, xgb_train, num_rounds, watchlist, early_stopping_rounds=5, verbose_eval=20)
    predict = model.predict(xgb_test, ntree_limit=model.best_ntree_limit)
    predict = pd.DataFrame(predict, columns=['shop_id'])
    predict['shop_id'] = predict['shop_id'].apply(lambda x: lbl.inverse_transform(int(x)))
    train_acc = accuracy_score(y_train, model.predict(xgb_train))
    return predict, train_acc


def lgb(x_train, y_train, x_test, best_iterations):
    y_train, num_class, lbl = label_encode(y_train)
    params = {
        'num_class': num_class,
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

    lgb_train = lightgbm.Dataset(x_train, y_train)
    bst = lightgbm.train(params, lgb_train, num_boost_round=150, verbose_eval=20, valid_sets=[lgb_train],
                         early_stopping_rounds=5)

    pred = bst.predict(x_test, num_iteration=bst.best_iteration)
    predict = []
    for x in pred:
        x = list(x)
        predict.append(x.index(sorted(x, reverse=True)[0]))
    predict = pd.DataFrame(predict, columns=['shop_id'])
    predict['shop_id'] = predict['shop_id'].apply(lambda x: lbl.inverse_transform(int(x)))
    # train_acc = accuracy_score(y_train, bst.predict(x_train))
    train_acc = 0
    return predict, train_acc



def nn(x_train, y_train, x_test):
    from keras.layers import Dense, Dropout, BatchNormalization
    from keras.optimizers import SGD, RMSprop
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.utils import np_utils
    from keras.regularizers import l2
    from keras.models import Sequential
    from keras.utils import np_utils
    y_train, num_class, lbl = label_encode(y_train)
    y_train = np_utils.to_categorical(y_train)

    clf = Sequential()
    clf.add(Dense(128, input_dim=x_train.shape[1], activation="relu", W_regularizer=l2()))
    # clf.add(SReLU())
    # clf.add(Dropout(0.2))
    clf.add(Dense(64, activation="relu", W_regularizer=l2()))
    # clf.add(SReLU())
    # clf.add(Dense(128, activation="relu", W_regularizer=l2()))
    # model.add(Dropout(0.2))
    clf.add(Dense(num_class, activation="softmax"))
    clf.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    reduce = ReduceLROnPlateau(min_lr=0.0002, factor=0.05)
    clf.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['acc', 'mae'])
    clf.fit(x_train, y_train,
            batch_size=640,
            nb_epoch=1000,
            validation_data=[x_train, y_train],
            callbacks=[early_stopping, reduce],
            verbose=2,
            shuffle=False)
    pred = clf.predict_proba(x_test, verbose=0)
    predict = []
    for x in pred:
        x = list(x)
        predict.append(x.index(sorted(x, reverse=True)[0]))
    predict = pd.DataFrame(predict, columns=['shop_id'])
    predict['shop_id'] = predict['shop_id'].apply(lambda x: lbl.inverse_transform(int(x)))
    train_acc = 0
    return predict, train_acc


def rf_prob(train, label, test):
    train = train.fillna(-999)
    test = test.fillna(-999)
    randomforest = RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=2017,
                                          max_features="auto", verbose=0)
    randomforest.fit(train, label)
    predict = randomforest.predict_proba(test)
    return predict


def et_prob(train, label, test):
    extratree = ExtraTreesClassifier(n_estimators=300, max_depth=None, max_features="auto", n_jobs=-1, random_state=2017,
                                     verbose=0)
    extratree.fit(train, label)
    predict = extratree.predict_proba(test)
    extratree = []
    del extratree
    return predict


def ada_prob(train, label, test):
    adaboost = AdaBoostClassifier(n_estimators=500, random_state=2017, learning_rate=0.001)
    adaboost.fit(train, label)
    predict = adaboost.predict_proba(test)
    return predict


def ovr_prob(train, label, test):
    train = train.fillna(-999)
    test = test.fillna(-999)
    n_estimators = 400
    max_depth = None
    est = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=2017,
                                 max_features="auto",
                                 verbose=0)
    ovr = OneVsRestClassifier(est, n_jobs=-1)
    ovr.fit(train, label)
    pred = ovr.predict_proba(test)
    return pred


def xgb_prob(x_train, y_train, x_test, num_class):
    params = {
        'objective': 'multi:softprob',
        'eta': 0.1,
        'max_depth': 5,
        'eval_metric': 'merror',
        'seed': 0,
        'missing': -999,
        'num_class': num_class,
        'silent': 1
    }
    xgb_train = xgboost.DMatrix(x_train, label=y_train)
    xgb_test = xgboost.DMatrix(x_test)
    watchlist = [(xgb_train, 'train'), (xgb_train, 'test')]
    num_rounds = 500
    model = xgboost.train(params, xgb_train, num_rounds, watchlist, early_stopping_rounds=5, verbose_eval=20)
    predict = model.predict(xgb_test, ntree_limit=model.best_ntree_limit)
    return predict


def lgb_prob(x_train, y_train, x_test, num_class, best_iterations):
    params = {
        'num_class': num_class,
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
        'num_iterations': best_iterations,
        'lambda_l1': [0.01],
        'lambda_l2': [1],
        'verbose': [0],
        'is_unbalance': [True]
    }
    lgb_train = lightgbm.Dataset(x_train, y_train)
    bst = lightgbm.train(params, lgb_train, verbose_eval=20, valid_sets=[lgb_train])

    predict = bst.predict(x_test, num_iteration=bst.best_iteration)
    return predict


def nn_prob(x_train, y_train, x_test, num_class):
    from keras.layers import Dense, Dropout, BatchNormalization
    from keras.optimizers import SGD, RMSprop
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.utils import np_utils
    from keras.regularizers import l2
    from keras.models import Sequential
    from keras.utils import np_utils

    y_train = np_utils.to_categorical(y_train)

    clf = Sequential()
    clf.add(Dense(128, input_dim=x_train.shape[1], activation="relu", W_regularizer=l2()))
    # clf.add(SReLU())
    # clf.add(Dropout(0.2))
    clf.add(Dense(128, activation="relu", W_regularizer=l2()))
    # clf.add(SReLU())
    # clf.add(Dense(128, activation="relu", W_regularizer=l2()))
    # model.add(Dropout(0.2))
    clf.add(Dense(num_class, activation="softmax"))
    clf.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    reduce = ReduceLROnPlateau(min_lr=0.0002, factor=0.05)
    clf.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['acc', 'mae'])
    clf.fit(x_train, y_train,
            batch_size=640,
            nb_epoch=1000,
            validation_data=[x_train, y_train],
            callbacks=[early_stopping, reduce],
            verbose=2,
            shuffle=False)
    predict = clf.predict_proba(x_test, verbose=0)
    return predict
