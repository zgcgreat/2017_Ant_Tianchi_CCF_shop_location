import pandas as pd
from collections import defaultdict

path = '../data_ori/'


def rule_train(train, topK_weights, connect_weight):
    # topK_weights = [5, 4, 3, 2, 1]
    # connect_weight = 5
    wifi_to_shop = defaultdict(lambda: defaultdict(lambda: 0))  # 默认字典嵌套，wifi_to_shop[wifi][shop]为wifi与shop的关联个数
    for line in train.values:
        wifi = sorted([wifi.split('|') for wifi in line[5].split(';')], key=lambda x: int(x[1]), reverse=True)[: len(topK_weights)]  # 按wifi信号强度排序
        for i, each_wifi in enumerate(wifi):
            if each_wifi[2] == 'true':
                wifi_to_shop[each_wifi[0]][line[1]] += connect_weight  # 单独设权
            else:
                wifi_to_shop[each_wifi[0]][line[1]] += topK_weights[i]  # each_wifi[0]表示wifi的bssid  line[1]表示shop_id

    return wifi_to_shop


# 验证
def rule_eval(train, result_topK, wifi_to_shop):
    right_count = 0
    # result_topK = 5  # 根据周围最强的k个wifi来预测
    for line in train.values:
        wifi = sorted([wifi.split('|') for wifi in line[5].split(';')], key=lambda x: int(x[1]), reverse=True)[
               : result_topK]  # 按wifi信号强度排序
        counter = defaultdict(lambda: 0)  # 统计每家店的得分
        for each_wifi in wifi:
            for k, v in wifi_to_shop[each_wifi[0]].items():
                counter[k] += v
        pred_one = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
        if pred_one == line[1]:
            right_count += 1
    acc = 1.0 * right_count / len(train)
    return acc


def rule_pred(test, wifi_to_shop):
    preds = []
    for line in test.values:
        index = 0
        while True:
            try:
                if index == 5:
                    pred_one = None  #
                    break
                wifi = sorted([wifi.split('|') for wifi in line[6].split(';')], key=lambda x: int(x[1]), reverse=True)[
                        index]  # 按wifi信号强度排序
                counter = defaultdict(lambda: 0)  # 统计每家店的得分
                for k, v in wifi_to_shop[wifi[0]].items():
                    counter[k] += v
                pred_one = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                break
            except:
                index += 1
        preds.append(pred_one)
    return preds


if __name__ == '__main__':
    train = pd.read_csv(path + 'train.csv')
    test = pd.read_csv(path + 'test.csv')
    # 参数搜索
    # topK_weight_list = [[5, 4, 3, 2, 1], [3, 2, 1], [1]]
    # for topK_weights in topK_weight_list:
    #     for connect_weight in [topK_weights[0], topK_weights[0]*2, topK_weights[0]/2.0]:
    #         wifi_to_shop = rule_train(train, topK_weights, connect_weight)
    #         result_topKs = [1, len(topK_weights), int(len(topK_weights)/2), 10]
    #         for result_topK in result_topKs:
    #             if result_topK == 0:
    #                 break
    #             acc = rule_eval(train,  result_topK, wifi_to_shop)
    #             print(topK_weights, connect_weight, result_topK, 'acc:', acc)

    # 预测
    topK_weights = [1]
    connect_weight = 1
    wifi_to_shop = rule_train(train, topK_weights, connect_weight)
    preds = rule_pred(test, wifi_to_shop)
    # print(preds)
    result = pd.DataFrame({'row_id': test.row_id, 'shop_id': preds})
    result = result.fillna('s_666')
    result.to_csv('../output/relu_result.csv', index=False)
