import numpy as np
# import gc
from torch.utils.data import DataLoader, Dataset
import csv
import scipy.sparse as sp
import pickle
import pandas as pd
import os

class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def search_data(sequence_length, num_of_batches, label_start_idx,
                num_for_predict, num_for_hour_prev, units, points_per_hour, Q):#, is_hour=False):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data

    num_of_batches: int, the number of batches will be used for training，例如要取前两周同一星期数的数据，那么这个参数就是2，如果去前6天的数据，这个参数就是6

    label_start_idx: int, the first index of predicting target

    num_for_predict: int,
                     the number of points will be predicted for each sample，即预测时间序列的长度，注意不是预测小时数，而是预测序列长度

    num_for_hour_prev: int, 预测时刻前多少个小时

    units: int, week: 7 * 24, day: 24, recent(hour): 1

    points_per_hour: int, number of points per hour, depends on data

    Q: int, sliding window size

    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    if units == 1:
        start_idx = label_start_idx - num_for_hour_prev * points_per_hour
        end_idx = label_start_idx
        x_idx.append((start_idx, end_idx))
        return x_idx

    for i in range(1, num_of_batches + 1):
        tmp_idx = label_start_idx - points_per_hour * units * i
        start_idx = tmp_idx - num_for_hour_prev * points_per_hour
        end_idx = tmp_idx + num_for_predict + Q
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_batches:
        return None

    return x_idx[::-1]

def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12, Q=3):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)

    num_of_weeks, num_of_days, num_of_hours: int

    label_start_idx: int, the first index of predicting target

    num_for_predict: int,
                     the number of points will be predicted for each sample，即预测时间序列的长度

    points_per_hour: int, default 12, number of points per hour

    Q: int, sliding window size

    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)

    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)

    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)

    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                               label_start_idx, num_for_predict, num_of_hours,
                               7 * 24, points_per_hour, Q)
    if not week_indices:
        return None

    day_indices = search_data(data_sequence.shape[0], num_of_days,
                              label_start_idx, num_for_predict, num_of_hours,
                              24, points_per_hour, Q)
    if not day_indices:
        return None

    hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                               label_start_idx, num_for_predict, num_of_hours,
                               1, points_per_hour, Q)#, True)
    if not hour_indices:
        return None

    week_sample = np.concatenate([np.expand_dims(data_sequence[i: j], axis=0)
                                  for i, j in week_indices], axis=0)
    day_sample = np.concatenate([np.expand_dims(data_sequence[i: j], axis=0)
                                 for i, j in day_indices], axis=0)
    hour_sample = np.concatenate([np.expand_dims(data_sequence[i: j], axis=0)
                                  for i, j in hour_indices], axis=0)
    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return week_sample, day_sample, hour_sample, target


def data_split(dataset, file_path, num_of_weeks, num_of_days,
                              num_of_hours, num_for_predict,
                              points_per_hour=12, Q=3):
    if Q > num_of_hours * points_per_hour:
        print("Wrong Q!")
        exit(-1)
    if 'pems' in dataset.lower():
        data_seq = np.load(file_path)['data'][:, :, 0].astype(np.int)#[:4000]
        data_seq = np.expand_dims(data_seq, axis=-1)
    else:
        print("wrong dataset!")
        exit(0)

    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour, Q)
        if not sample:
            continue

        week_sample, day_sample, hour_sample, y = sample
        all_samples.append((
            np.expand_dims(week_sample, axis=0),#.transpose((0, 2, 3, 1)),
            np.expand_dims(day_sample, axis=0),#.transpose((0, 2, 3, 1)),
            np.expand_dims(hour_sample, axis=0),#.transpose((0, 2, 3, 1)),
            np.expand_dims(y, axis=0),#.transpose((0, 2, 3, 1))[:, :, 0, :]
        ))

    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    train_val_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line2])]
    test_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]
    train_val_week, train_val_day, train_val_hour, train_val_y = train_val_set
    train_val_data = {'week': train_val_week, 'day': train_val_day, 'hour': train_val_hour, 'y': train_val_y}
    test_week, test_day, test_hour, test_y = test_set
    test_data = {'week': test_week, 'day': test_day, 'hour': test_hour, 'y': test_y}

    scaler = StandardScaler(mean=train_val_hour.mean(),
                            std=train_val_hour.std())

    train_val_data['hour'] = scaler.transform(train_val_data['hour'])
    train_val_data['day'] = scaler.transform(train_val_data['day'])
    train_val_data['week'] = scaler.transform(train_val_data['week'])
    train_val_data['y'] = scaler.transform(train_val_data['y'])

    test_data['hour'] = scaler.transform(test_data['hour'])
    test_data['day'] = scaler.transform(test_data['day'])
    test_data['week'] = scaler.transform(test_data['week'])

    train_data = {'week': train_val_data['week'][:split_line1, :, :], 'day': train_val_data['day'][:split_line1, :, :],
                  'hour': train_val_data['hour'][:split_line1, :, :], 'y': train_val_data['y'][:split_line1, :, :]}
    val_data = {'week': train_val_data['week'][split_line1:split_line2, :, :], 'day': train_val_data['day'][split_line1:split_line2, :, :],
                'hour': train_val_data['hour'][split_line1:split_line2, :, :], 'y': train_val_data['y'][split_line1:split_line2, :, :]}

    return scaler, train_data, val_data, test_data

class FlowDataset(Dataset):
    def __init__(self, data):
        self.week, self.day, self.hour, self.y = data['week'], data['day'], data['hour'], data['y']

    def __len__(self):
        return self.week.shape[0]

    def __getitem__(self, idx):
        week_sample, day_sample, hour_sample, y = self.week[idx], self.day[idx], self.hour[idx], self.y[idx]
        return week_sample, day_sample, hour_sample, y

def get_dateloader(data, batch_size):
    dataset = FlowDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(
        np.float32).todense()

def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def pems_load_pickle(pickle_file, node_num):
    with open(pickle_file, 'r') as f:
        reader = csv.reader(f)
        _ = f.__next__()
        edges = []
        for i in reader:
            try:
                edges.append((int(i[0]), int(i[1]), float(i[2])))
            except:
                continue

    A = np.zeros((node_num, node_num), dtype=np.float32)
    cost_std = np.std([each[2] for each in edges], ddof=1)
    for i, j, cost in edges:
        A[i, j] = np.exp(-(cost / cost_std) ** 2)
    return A

def pems03_load_pickle(pickle_file, node_num):
    node_set_file = pickle_file[:-3] + 'txt'
    node_idx = 0
    node_d = {}
    with open(node_set_file, 'r') as f:
        for line in f:
            try:
                node_d[int(line.strip())] = node_idx
                node_idx += 1
            except:
                continue

    with open(pickle_file, 'r') as f:
        reader = csv.reader(f)
        _ = f.__next__()
        edges = []
        for i in reader:
            try:
                edges.append((int(i[0]), int(i[1]), float(i[2])))
            except:
                continue

    A = np.zeros((node_num, node_num), dtype=np.float32)
    cost_std = np.std([each[2] for each in edges], ddof=1)
    for i, j, cost in edges:
        A[node_d[i], node_d[j]] = np.exp(-(cost / cost_std) ** 2)
    return A

def load_adj(dataset, pkl_filename, node_num, sym_graph=False):
    dataset = dataset.lower()
    if dataset == "pems03":
        adj_mx = pems03_load_pickle(pkl_filename, node_num)
    else:
        adj_mx = pems_load_pickle(pkl_filename, node_num)
    if sym_graph:
        adj_mx += np.eye(node_num)
        return [sym_adj(adj_mx)]
    return [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
