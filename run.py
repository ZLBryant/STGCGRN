import argparse
import torch
import os
import numpy as np
import random
from data_util import data_split, get_dateloader, load_adj
import gc
from models import STGCGRN
from model_utils import model_train, model_test
import csv
import json
import configparser
import pandas as pd

random.seed(0)

def args_init(dataset):
    config_file = None
    if dataset in ['pems03', 'PEMS03']:
        config_file = "conf/PEMS03.conf"
    elif dataset in ['pems04', 'PEMS04']:
        config_file = "conf/PEMS04.conf"
    elif dataset in ['pems07', 'PEMS07']:
        config_file = "conf/PEMS07.conf"
    elif dataset in ['pems08', 'PEMS08']:
        config_file = "conf/PEMS08.conf"
    else:
        print("wrong dataset!")
        exit(0)
    config = configparser.ConfigParser()
    config.read(config_file, encoding="utf-8")
    parser = argparse.ArgumentParser(description=config['data']['dataset_name'])
    parser.add_argument("--dataset", type=str, default=config['data']['dataset_name'])
    parser.add_argument("--model_kind", type=str, default="stdgru")#no_pre/no_adp/no_period/no_window
    parser.add_argument("--dataset_path", type=str, default=config['data']['dataset_path'])
    parser.add_argument("--adjdata", type=str, default=config['data']['adjdata'])
    parser.add_argument("--node_num", type=int, default=config['data']['node_num'])
    parser.add_argument("--points_per_hour", type=int, default=config['data']['points_per_hour'])
    parser.add_argument("--num_for_predict", type=int, default=config['data']['num_for_predict'])
    parser.add_argument("--layer_num", type=int, default=config['model']['layer_num'], help='the number of gru leyer')
    parser.add_argument("--input_dim", type=int, default=config['model']['input_dim'])
    parser.add_argument("--hidden_dim", type=int, default=config['model']['hidden_dim'])
    parser.add_argument("--output_dim", type=int, default=config['model']['output_dim'])
    parser.add_argument("--node_emb_dim", type=int, default=config['model']['node_emb_dim'])
    parser.add_argument("--agcn_head_num", type=int, default=config['model']['agcn_head_num'])
    parser.add_argument("--w_pre", type=float, default=config['model']['w_pre'])
    parser.add_argument("--w_adp", type=float, default=config['model']['w_adp'])
    parser.add_argument("--lr", type=float, default=config['train']['lr'])
    parser.add_argument("--batch_size", type=int, default=config['train']['batch_size'])
    parser.add_argument("--dropout", type=float, default=config['train']['dropout'])
    parser.add_argument("--num_of_weeks", type=int, default=config['train']['num_of_weeks'])
    parser.add_argument("--num_of_days", type=int, default=config['train']['num_of_days'])
    parser.add_argument("--num_of_hours", type=int, default=config['train']['num_of_hours'])
    parser.add_argument("--Q", type=int, default=config['train']['Q'], help="Sliding window size: 2*Q+1, where Q should not be greater than num_of_hours*points_per_hour")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    return args

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def model_main(args, seeds, out_dir):
    args.device = ('cuda' if torch.cuda.is_available() else 'cpu')
    scaler, train_data, val_data, test_data = data_split(args.dataset, args.dataset_path, args.num_of_weeks, args.num_of_days,
                                                              args.num_of_hours, args.num_for_predict, args.points_per_hour, args.Q)
    train_dataloader = get_dateloader(train_data, args.batch_size)
    val_dataloader = get_dateloader(val_data, args.batch_size)
    test_dataloader = get_dateloader(test_data, args.batch_size)
    del train_data, val_data, test_data
    gc.collect()

    predefined_A = load_adj(args.dataset, args.adjdata, args.node_num, True)
    predefined_A = [torch.tensor(adj).to(args.device) for adj in predefined_A]

    first = True
    val_losses, metrics = [], {"total_RMSE": [], "total_MAE": [], "total_MAPE": [], "total_MSE": [], "MAE": [], "MAPE": [], "MSE": [], "RMSE": []}
    for seed in seeds:
        setup_seed(seed)
        cur_out_dir = os.path.join(out_dir, str(seed))
        if not os.path.exists(cur_out_dir):
            os.makedirs(cur_out_dir)

        model = STGCGRN(args, predefined_A).to(args.device)
        model_save_path = os.path.join(cur_out_dir, "model" + ".pkl")
        torch.save(model, model_save_path)
        val_loss = model_train(model, train_dataloader, val_dataloader, args, model_save_path)
        model = torch.load(model_save_path)
        test_metric = model_test(model, test_dataloader, args.device, scaler)

        for k, v in test_metric.items():
            metrics[k].append(np.array(v))

        val_losses.append(val_loss)

        with open(out_dir + '.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            if first:
                columns = ['seed', 'lr', 'dropout', 'batch_size', 'layer_num',
                           'w_pre', 'w_adp', 'agcn_head_num', 'val_loss', 'blank'] + list(
                    test_metric.keys())
                writer.writerow(columns)
                first = False
            row = [seed, args.lr, args.dropout, args.batch_size, args.layer_num,
                   args.w_pre, args.w_adp, args.agcn_head_num, val_loss, ' '] + list(test_metric.values())
            writer.writerow(row)

    del train_dataloader, val_dataloader, test_dataloader
    gc.collect()

    csv_dict = pd.read_csv(out_dir + '.csv').to_dict()
    metrics = {}
    seed_num = len(csv_dict['seed'])
    for k in ["total_MAE", "total_MAPE", "total_RMSE", "total_MSE", "MAE", "MAPE", "RMSE", "MSE"]:
        v = csv_dict[k].values()
        if not k.startswith("total_"):
            v_list = []
            for each in v:
                v_list.append(np.array(list(map(float, each[1:-1].split(",")))))
            metrics[k] = (sum(v_list) / seed_num).tolist()
        else:
            metrics[k] = (sum(v) / seed_num)
    metric_path = os.path.join(out_dir, 'metric.json')
    metrics['val_loss'] = sum(csv_dict['val_loss'].values()) / seed_num
    with open(metric_path, 'w') as f:
        f.write(json.dumps(metrics, ensure_ascii=False, indent=2))
    #return metrics['val_loss']

if __name__ == "__main__":
    args = args_init('pems08')
    seeds = random.sample(range(0, 2147483647), 5)
    out_dir = os.path.join('output', args.dataset, args.model_kind)
    if 'no_adp' in args.model_kind:
        args.w_pre, args.w_adp = 1., 0.
    elif 'no_pre' in args.model_kind:
        args.w_pre, args.w_adp = 0., 1.
    else:
        args.w_pre, args.w_adp = 0.1, 0.9
    model_main(args, seeds, out_dir)
