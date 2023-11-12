import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def get_tgt_mask(biased_mask, T):
    biased_mask = biased_mask[:, :T, :T].clone().detach()
    mask_last = torch.zeros(T, T)
    mask_last[-1,:-1] = 1
    mask = torch.eye(T, T)
    mask = mask + mask_last
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    biased_mask = mask.unsqueeze(0) + biased_mask
    return biased_mask

# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


def init_biased_mask2(n_head, window_size, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))

    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(window_size, max_seq_len)
    for i in range(window_size):
        alibi[i, :max_seq_len - window_size + i +1] = bias[window_size-i-1:]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = torch.triu(torch.ones(window_size, max_seq_len))  == 1
    mask = torch.flip(mask, dims = [0, 1])

    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(device,  T, S):
    mask = torch.ones(T, S)
    # if dataset == "BIWI":
    #     for i in range(T):
    #         mask[i, i*2:i*2+2] = 0
    # elif dataset == "vocaset":
    #     for i in range(T):
    #         mask[i, i] = 0
    for i in range(T):
        mask[i, i*2:i*2+2] = 0
    return (mask==1).to(device=device)

def enc_dec_mask2(device, T, S):
    mask = torch.ones(T, S)
    # if dataset == "BIWI":
    #     for i in range(T):
    #         mask[i, i*2:i*2+2] = 0
    # elif dataset == "vocaset":
    #     for i in range(T):
    #         mask[i, i] = 0
    for i in range(T):
        mask[i, i] = 0
    return (mask==1).to(device=device)


def enc_dec_mask3(device, T, S):
    mask = torch.ones(T, S)
    # if dataset == "BIWI":
    #     for i in range(T):
    #         mask[i, i*2:i*2+2] = 0
    # elif dataset == "vocaset":
    #     for i in range(T):
    #         mask[i, i] = 0
    for i in range(T):
        mask[i, :S-T+i+1] = 0
    return (mask==1).to(device=device)



# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=751):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=True):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

