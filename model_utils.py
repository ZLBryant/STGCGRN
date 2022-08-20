import torch
from torch import optim
import numpy as np

def compute_val_loss(model, val_dataloader, device):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for batch_idx, batch_data in enumerate(val_dataloader):
            [week_sample, day_sample, hour_sample, target] = batch_data
            week_sample, day_sample, hour_sample, target = week_sample.to(torch.float32).to(
                device), day_sample.to(torch.float32).to(device), hour_sample.to(
                torch.float32).to(
                device), target.to(torch.float32).to(device)
            output = model(week_sample, day_sample, hour_sample)
            predict = output
            loss = masked_mae(predict, target, 0.0)
            val_loss += loss.detach().cpu().numpy()
    val_loss /= len(val_dataloader)
    return val_loss

def model_train(model, train_dataloader, val_dataloader, args, model_save_path):
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    train_loss_list, val_loss_list = [], []
    for cur_epoch in range(50):
        print("cur_epoch:", cur_epoch)
        val_loss = compute_val_loss(model, val_dataloader, args.device)
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(model, model_save_path)
        model.train()
        train_loss = 0
        for batch_idx, batch_data in enumerate(train_dataloader):
            [week_sample, day_sample, hour_sample, target] = batch_data
            week_sample, day_sample, hour_sample, target = week_sample.to(torch.float32).to(
                args.device), day_sample.to(torch.float32).to(args.device), hour_sample.to(torch.float32).to(
                args.device), target.to(torch.float32).to(args.device)
            [batch_size, _, node_num, out_dim] = target.shape

            optimizer.zero_grad()
            output = model(week_sample, day_sample, hour_sample,
                           target.transpose(1, 2).reshape(batch_size * node_num, -1, out_dim))
            predict = output
            loss = masked_mae(predict, target, 0.0)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().cpu().numpy()
        train_loss = train_loss / len(train_dataloader)
        train_loss_list.append(train_loss)

    lr = args.lr * 0.1
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for cur_epoch in range(25):
        model.train()
        train_loss = 0
        for batch_idx, batch_data in enumerate(train_dataloader):
            [week_sample, day_sample, hour_sample, target] = batch_data
            week_sample, day_sample, hour_sample, target = week_sample.to(torch.float32).to(
                args.device), day_sample.to(torch.float32).to(args.device), hour_sample.to(torch.float32).to(
                args.device), target.to(torch.float32).to(args.device)

            optimizer.zero_grad()
            output = model(week_sample, day_sample, hour_sample, fine_tuning=True)
            predict = output
            loss = masked_mae(predict, target, 0.0)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().cpu().numpy()
        train_loss = train_loss / len(train_dataloader)
        train_loss_list.append(train_loss)
        val_loss = compute_val_loss(model, val_dataloader, args.device)
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(model, model_save_path)
    ################################################
    return best_val_loss

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels,
                                 null_val=null_val))

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def evaluate(pred, real):
    res = {}
    res["MSE"] = masked_mse(pred, real, 0.0).item()
    res["MAE"] = masked_mae(pred, real, 0.0).item()
    res["MAPE"] = masked_mape(pred, real, 0.0).item()
    res["RMSE"] = masked_rmse(pred, real, 0.0).item()
    return res

def model_test(model, test_dataloader, device, scaler):
    model.eval()
    predicts, targets = [], []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            [week_sample, day_sample, hour_sample, target] = batch_data
            week_sample, day_sample, hour_sample, target = week_sample.to(torch.float32).to(
                device), day_sample.to(torch.float32).to(device), hour_sample.to(torch.float32).to(
                device), target.to(torch.float32).to(device)
            output = model(week_sample, day_sample, hour_sample)
            predict = scaler.inverse_transform(output)
            predicts.append(predict)
            targets.append(target)
    predicts = torch.cat(predicts, dim=0)
    targets = torch.cat(targets, dim=0)
    metric = {"MAE": [], "MAPE": [], "MSE": [], "RMSE": []}
    for i in range(predicts.shape[1]):
        cur_metric = evaluate(predicts[:, i, :], targets[:, i, :])
        for k, v in cur_metric.items():
            metric[k].append(v)
    total_res = evaluate(predicts, targets)
    metric["total_RMSE"] = total_res["RMSE"]
    metric["total_MAE"] = total_res["MAE"]
    metric["total_MAPE"] = total_res["MAPE"]
    metric["total_MSE"] = total_res["MSE"]
    return metric
