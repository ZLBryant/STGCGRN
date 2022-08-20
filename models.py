import torch.nn as nn
import torch
import torch.nn.functional as F
import math
# import numpy as np

class gconv_hyper(nn.Module):
    def __init__(self):
        super(gconv_hyper, self).__init__()

    def forward(self, x, A):
        if len(A.shape) == 2:
            try:
                x = torch.einsum('wv,bvc->bwc', (A, x))  # x:batch_size * node_num * hidden_dim, A:node_num * node_num
            except:
                print("x shape:", x.shape)
                print("A shape:", A.shape)
                exit(-1)
        elif len(A.shape) == 3:
            if x.shape[0] == A.shape[0] and len(x.shape) == len(A.shape):
                x = torch.einsum('bwv,bvc->bwc',
                                 (A, x))  # x:batch_size * node_num * hidden_dim, A:batch_size * node_num * node_num
            else:
                batch_size, head_num = x.shape[0], A.shape[0]
                A = A.unsqueeze(0).repeat(batch_size, 1, 1, 1)
                x = torch.einsum('bnwv,bnvc->bnwc', (
                A, x))  # x:batch_size * 1 * node_num * hidden_dim, A:1 * head_num * node_num * node_num
        elif len(A.shape) == 4:
            x = torch.einsum('bnwv,bnvc->bnwc', (
            A, x))  # x:batch_size * node_num * hidden_dim, A:batch_size * head_num * node_num * node_num
        return x.contiguous()

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, gdep=2):
        super(GCNLayer, self).__init__()
        self.gdep = gdep
        self.gconv_preA = gconv_hyper()
        self.mlp = nn.Linear((self.gdep + 1) * in_dim, out_dim)

    def forward(self, adj, x):#x: batch_size * node_num * (hidden_size + feature_dim)
        h = x
        out = [h]
        for _ in range(self.gdep):
            h = self.gconv_preA(h, adj)
            out.append(h)
        ho = torch.cat(out, dim=-1)
        ho = self.mlp(ho)
        return ho

class GCRNCell(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCRNCell, self).__init__()
        self.gz = GCNLayer(in_dim, out_dim)
        self.gr = GCNLayer(in_dim, out_dim)
        self.gc = GCNLayer(in_dim, out_dim)

    def self_attn_func(self, x):
        # x.shape: batch_size * node_num * hidden_dim
        x = x.transpose(0, 1)
        x_attn_out, _ = self.self_attn(x, x, x)
        x_attn_out = x_attn_out.transpose(0, 1)
        return x_attn_out

    def gcn_forward(self, x_hidden, predefine_A, add_weight, gcn_layers, E1=None, E2=None):
        # x_hidden = torch.cat((input, hidden_state), dim=-1)
        res = 0
        if add_weight['w_pre'] > 0.00001:
            predefine_gcn = sum([gcn_layers(each, x_hidden) for each in predefine_A])
            res += add_weight['w_pre'] * predefine_gcn

        if add_weight['w_adp'] > 0.00001:
            [_, head_num, _] = E1.shape
            sqrt_d = math.sqrt(E1.shape[-1])
            A = F.relu(torch.einsum('nhc,nvc->nhv', (E1.transpose(0, 1), E2.transpose(0, 1)))) / sqrt_d
            A = F.softmax(A, dim=-1)
            agcn_res = gcn_layers(A, x_hidden.unsqueeze(1).repeat(1, head_num, 1, 1)).mean(1)
            res += add_weight['w_adp'] * agcn_res

        return res

    def forward(self, hidden_state, predefine_A, add_weight, E1=None, E2=None):
        '''
        :param input: batch_size * node_num * 1
        :param hidden_state: batch_size * node_num * hidden_dim
        :param predefine_A: node_num * node_num
        :param E1: node_num * agcn_head_num * (node_emb_dim // agcn_head_num)
        :param E2:
        :param batch_size:
        :return:
        '''
        z = F.sigmoid(self.gcn_forward(hidden_state, predefine_A, add_weight, self.gz, E1, E2)) #batch_size * node_num * hidden_dim
        r = F.sigmoid(self.gcn_forward(hidden_state, predefine_A, add_weight, self.gr, E1, E2)) #batch_size * node_num * hidden_dim
        c = F.tanh(self.gcn_forward(torch.mul(r, hidden_state), predefine_A, add_weight, self.gc, E1, E2))
        hidden_state = torch.mul(1 - z, hidden_state) + torch.mul(z, c)
        return hidden_state

class AttnLayer(nn.Module):
    def __init__(self, hidden_dim, attn_dim):
        super(AttnLayer, self).__init__()
        self.Wx = nn.Linear(in_features=hidden_dim, out_features=attn_dim, bias=True)
        self.Wh = nn.Linear(in_features=hidden_dim, out_features=attn_dim, bias=True)
        self.v = nn.Linear(in_features=attn_dim, out_features=1, bias=False)

    def forward(self, cur_h, history_h):
        # cur_h: (batch_size*node_num, hidden_dim)
        # history_h: (batch_size*node_num, day_num+week_num, window_size, hidden_dim)
        history_h_project = self.Wh(history_h)# batch_size * (day_num + week_num) * seq_len * hidden_dim
        cur_h_project = self.Wx(cur_h)[:, None, None, :]
        score = self.v(torch.tanh(history_h_project + cur_h_project)).squeeze(-1)# score：batch_size * (day + week) * seq_len
        history_h_flat = history_h.reshape(history_h.shape[0], -1, history_h.shape[-1])
        score_flat = score.reshape(score.shape[0], -1).unsqueeze(-1)
        attn_score = F.softmax(score_flat, 1)
        attn_h = (attn_score * history_h_flat).sum(1)
        out = cur_h + attn_h
        # out = torch.cat((cur_h, attn_h), 1)
        return out

class STGCGRN(nn.Module):
    def __init__(self, args, predefine_A=None):
        super(STGCGRN, self).__init__()
        self.num_for_predict = args.num_for_predict
        self.layer_num = args.layer_num

        self.predefine_A = predefine_A
        if "no_adp_pre" in args.model_kind:
            node_num = predefine_A[0].shape[0]
            self.predefine_A = [torch.eye(node_num).to(args.device)]

        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim

        self.short_gru = nn.GRU(args.input_dim,
                                self.hidden_dim,
                                args.layer_num,
                                batch_first=True,
                                bidirectional=False)
        self.long_gru = nn.GRU(args.input_dim,
                               self.hidden_dim,
                               args.layer_num,
                               batch_first=True,
                               bidirectional=False)

        self.decoder = nn.GRUCell(self.output_dim, self.hidden_dim)

        self.add_weight = {"w_pre": args.w_pre, "w_adp": args.w_adp}

        self.spatial_dependency_layer = GCRNCell(self.hidden_dim, self.hidden_dim)

        # self-adaptive graph
        if self.add_weight['w_adp'] > 0.00001:
            self.head_num = args.agcn_head_num
            self.E1 = nn.Parameter(
                torch.randn(args.node_num, args.agcn_head_num, args.node_emb_dim // args.agcn_head_num),
                requires_grad=True)
            self.E2 = nn.Parameter(
                torch.randn(args.node_num, args.agcn_head_num, args.node_emb_dim // args.agcn_head_num),
                requires_grad=True)
        else:
            self.E1, self.E2 = None, None

        # attn_layer
        attn_dim = self.hidden_dim // 2
        self.attn_layers = AttnLayer(self.hidden_dim, attn_dim)

        self.output_dim = args.output_dim
        self.fc_layer = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(in_features=self.hidden_dim, out_features=args.output_dim)
        )
        self.Q = args.Q
        self.device = args.device
        self.model_kind = args.model_kind
        self.agcn_head_num = args.agcn_head_num

    def generate_window_mask(self, idx, mask_len):
        mask = torch.zeros(mask_len)
        mask[idx-self.Q: idx+self.Q+1] = 1
        # mask = mask.masked_fill(mask == 0, float("-inf"))#注意不能再这里进行替换，后续操作如果一个负数乘以-inf，会出现inf，然后softmax会变成nan
        return mask.to(self.device)

    def node_dependency(self, hidden_state):
        # batch_size = hidden_state.shape[0]
        # cur_E1, cur_E2 = self.E1.unsqueeze(0).repeat(batch_size, 1, 1, 1), self.E2.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        res = self.spatial_dependency_layer(hidden_state, self.predefine_A, self.add_weight, self.E1, self.E2)
        return res

    def forward(self, week_sample, day_sample, hour_sample, target=None, fine_tuning=False):
        hour_sample = hour_sample.squeeze(1)
        [batch_size, hour_seq_len, node_num, input_dim] = hour_sample.shape
        hour_sample = hour_sample.transpose(1, 2).reshape(-1, hour_seq_len, input_dim)
        cur_hs, _ = self.short_gru(hour_sample)

        history_sample = torch.cat((week_sample, day_sample), dim=1)
        history_seq_len = history_sample.shape[2]
        history_hs = []
        for i in range(history_sample.shape[1]):
            cur_history_hs, _ = self.long_gru(history_sample[:, i, :, :].transpose(1, 2).reshape(-1, history_seq_len, input_dim))
            history_hs.append(cur_history_hs.unsqueeze(1))
        history_hs = torch.cat(history_hs, dim=1)

        predict = []
        decoder_input = torch.zeros((batch_size*node_num, self.output_dim), device=self.device)
        hidden_state = cur_hs[:, -1, :]#(batch_size*node_num, hidden_dim)

        for i in range(self.num_for_predict):
            hidden_state = self.decoder(decoder_input, hidden_state)
            if "no_period" in self.model_kind:
                attn_res = hidden_state.reshape(batch_size, node_num, -1)
            else:
                if "no_window" in self.model_kind:
                    cur_history_hs = history_hs
                else:
                    idx = i + hour_seq_len
                    cur_history_hs = history_hs[:, :, idx - self.Q: idx + self.Q + 1, :]
                attn_res = self.attn_layers(hidden_state, cur_history_hs).reshape(batch_size, node_num, -1)
            node_dependency_res = self.node_dependency(attn_res)  # (batch_size, node_num, hidden_dim)
            cur_predict = self.fc_layer(node_dependency_res)  # (batch_size, node_num, 1)
            predict.append(cur_predict.unsqueeze(1))
            decoder_input = cur_predict.reshape(batch_size * node_num, -1)

            if self.training and not fine_tuning:
                decoder_input = target[:, i]#.unsqueeze(-1)

        return torch.cat(predict, 1)
