import pandas as pd
import numpy as np
import json
import sys
import os
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
from libcity.evaluator.utils import evaluate_model


config = {
    'model': 'HA',
    'lag': [36, 1],
    'weight': [0.1, 0.9],
    'n_sample': [3, 3],
    'dataset': 'Ocean_sensor',
    'train_rate': 0.7,
    'data_col': 'temp',
    'eval_rate': 0.1,
    'input_window': 12,
    'output_windows': 3,
    'metrics': ['MAE', 'MAPE', 'MSE', 'RMSE', 'masked_MAE',
                'masked_MAPE', 'masked_MSE', 'masked_RMSE', 'R2', 'EVAR']
}


def get_data(dataset):
    # path
    path = root_path + '/raw_data/' + dataset + '/'
    config_path = path + 'config.json'
    dyna_path = path + dataset + '.dyna'
    geo_path = path + dataset + '.geo'

    # read config
    with open(config_path, 'r') as f:
        json_obj = json.load(f)
        for key in json_obj:
            if key not in config:
                config[key] = json_obj[key]

    # read geo
    geo_file = pd.read_csv(geo_path)
    geo_ids = list(geo_file['geo_id'])

    # read dyna
    dyna_file = pd.read_csv(dyna_path)
    data_col = config.get('data_col', '')
    if data_col != '':  # 根据指定的列加载数据集
        if isinstance(data_col, list):
            data_col = data_col.copy()
        else:  # str
            data_col = [data_col].copy()
        data_col.insert(0, 'time')
        data_col.insert(1, 'entity_id')
        dyna_file = dyna_file[data_col]
    else:  # 不指定则加载所有列
        dyna_file = dyna_file[dyna_file.columns[2:]]  # 从time列开始所有列

    # 求时间序列
    time_slots = list(dyna_file['time'][:int(dyna_file.shape[0] / len(geo_ids))])

    # 转3-d数组
    feature_dim = len(dyna_file.columns) - 2
    df = dyna_file[dyna_file.columns[-feature_dim:]]
    len_time = len(time_slots)
    data = []
    for i in range(0, df.shape[0], len_time):
        data.append(df[i:i + len_time].values)
    data = np.array(data, dtype=float)  # (N, T, F)
    data = data.swapaxes(0, 1)  # (T, N, F)
    return data


def historical_average(data):
    t, n, f = data.shape
    train_rate = config.get('train_rate', 0.7)
    eval_rate = config.get('eval_rate', 0.1)
    output_window = config.get('output_window', 3)
    lag = config.get('lag', 7 * 24 * 12)
    weight = config.get('weight', 1.0)
    n_sample = config.get('n_sample', 4)
    if isinstance(lag, int):
        lag = [lag]
    if isinstance(weight, int) or isinstance(weight, float):
        weight = [weight]
    if isinstance(n_sample, int):
        n_sample = [n_sample]
    assert sum(weight) == 1
    assert int(t * (train_rate + eval_rate)) > max(np.array(n_sample) * np.array(lag))

    y_true = []
    y_pred = []
    for i in range(int(t * (train_rate + eval_rate)), t):
        # y_true
        y_true.append(data[i, :, :])  # (N, F)
        # y_pred
        y_pred_i = 0
        for j in range(len(lag)):
            # 隔lag[j]时间步采样n_sample[j]次, 得到(n_sample[j], N, F)取平均值得到(N, F), 最后用weight[j]加权
            y_pred_i += weight[j] * np.mean(data[i - n_sample[j] * lag[j]:i:lag[j], :, :], axis=0)
        y_pred.append(y_pred_i)  # (N, F)

    y_pred = np.array(y_pred)  # (test_size, N, F)
    y_true = np.array(y_true)  # (test_size, N, F)
    y_pred = np.expand_dims(y_pred, axis=1)  # (test_size, 1, N, F)
    y_true = np.expand_dims(y_true, axis=1)  # (test_size, 1, N, F)
    y_pred = np.repeat(y_pred, output_window, axis=1)  # (test_size, out, N, F)
    y_true = np.repeat(y_true, output_window, axis=1)  # (test_size, out, N, F)
    return y_pred, y_true


def main():
    print(config)
    data = get_data(config.get('dataset', ''))
    y_pred, y_true = historical_average(data)
    evaluate_model(y_pred=y_pred, y_true=y_true, metrics=config['metrics'],
                   path=config['model']+'_'+config['dataset']+'_metrics.csv')


if __name__ == '__main__':
    main()
