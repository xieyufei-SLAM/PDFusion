__all__ = ['PatchTST_backbone']

from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

# from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN


# Cell
class PatchTST_backbone(nn.Module):
    def __init__(self, c_in: int, context_window: int, target_window: int, patch_len: int, stride: int,
                 max_seq_len: Optional[int] = 1024,
                 n_layers: int = 3, d_model=128, n_heads=16, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256, norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, fc_dropout: float = 0., head_dropout=0, padding_patch=None,
                 pretrain_head: bool = False, head_type='flatten', individual=False, revin=True, affine=True,
                 subtract_last=False,
                 verbose: bool = False, **kwargs):

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))  # 在尾部复制并填充
            patch_num += 1

        # Backbone
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                    n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                    attn_dropout=attn_dropout, dropout=dropout, act=act,
                                    key_padding_mask=key_padding_mask, padding_var=padding_var,
                                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                    store_attn=store_attn,
                                    pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head:
            self.head = self.create_pretrain_head(self.head_nf, c_in,
                                                  fc_dropout)  # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten':
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                     head_dropout=head_dropout)
        # from layers.Embed import DataEmbedding_Feature_Time
        # self.Data_embedding = DataEmbedding_Feature_Time(c_in,d_model)

    def forward(self, z):  # z: [bs x nvars x seq_len]
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        # do patching z:[128,7,96] --> [128, 7, 104]
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)  # [128, 7, 104]

        # z = self.Data_embedding(z)
        # 转换为patch的关键步骤，相当于只有卷没有积
        z = z.unfold(dimension=-1, size=self.patch_len,
                     step=self.stride)  # [128,7,12,16]                    # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0, 1, 3,
                      2)  # [128,7,16,12]                                                              # z: [bs x nvars x patch_len x patch_num]

        # model [128,7,16,12]-->[128, 7, 512, 12]
        z = self.backbone(z)  # torch.Size([128, 7, 512, 12] # z: [bs x nvars x d_model x patch_num]
        # [128, 7, 512, 12] --> [128, 7, 512*12] --> [128, 7, 96]
        z = self.head(z)  # 先flatten倒数两个维度 再进行linear:torch.Size([128, 7, 96]  # z: [bs x nvars x target_window]

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')  # 反归一化
            z = z.permute(0, 2, 1)
        return z

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                             nn.Conv1d(head_nf, vars, 1)
                             )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                  attn_dropout=attn_dropout, dropout=dropout,
                                  pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                  store_attn=store_attn)

    def forward(self, x) -> Tensor:  # x: [bs x nvars x patch_len x patch_num]

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3,
                      2)  # [128, 7, 12, 16]                                                    # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(
            x)  # [128, 7, 12, 512]                                               # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2],
                              x.shape[3]))  # [128*7, 12, 512]       # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(
            u + self.W_pos)  # [128*7, 12, 512]                                          # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(
            u)  # [128*7, 12, 512]                                                       # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (
        -1, n_vars, z.shape[-2], z.shape[-1]))  # [128, 7, 12, 512]              # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0, 1, 3,
                      2)  # [128, 7, 512, 12]                                                 # z: [bs x nvars x d_model x patch_num]

        return z

    # Cell


class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList(
            [TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                             attn_dropout=attn_dropout, dropout=dropout,
                             activation=activation, res_attention=res_attention,
                             pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        output = src
        scores = None
        freq_scores = None
        # 输入大小：[128*7, 12, 512]
        if self.res_attention:
            for mod in self.layers: output, scores, freq_scores = mod(output, prev=scores, prev_freq=freq_scores,
                                                                      key_padding_mask=key_padding_mask,
                                                                      attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False,
                 pre_norm=False):  # q_len:12
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, q_len, q_len, d_k, d_v, attn_dropout=attn_dropout,
                                             proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None, prev_freq: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores, freq_attn, freq_scores = self.self_attn(src, src, src, prev, prev_freq,
                                                                        key_padding_mask=key_padding_mask,
                                                                        attn_mask=attn_mask)
        else:
            src2, attn, freq_attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask,
                                                   attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
            self.freq_attn = freq_attn
        ## Add & Norm
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores, freq_scores
        else:
            return src


# TODO 动刀改进为趋势信息交互注意力机制(包括FFT和时域频域交互序列)
class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, len_q, len_kv, d_k=None, d_v=None, res_attention=False, attn_dropout=0.,
                 proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention

        # TODO 使用时频交互注意力替代DotProdcut
        # self.interactive_attn = InteractiveTimeFeqAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)
        self.ITFA = InteractiveTimeFeqAttention(d_model, n_heads, len_q, len_kv, attn_dropout=attn_dropout,
                                                res_attention=self.res_attention, lsa=lsa)
        # self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                prev_freq: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads) # [896, 8, 12, 64]
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,
                                                                         2)  # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3,
                                                                       1)  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_socres_freqw, attn_scores, attn_socres_freq = self.ITFA(q_s, k_s, v_s, prev=prev,
                                                                                               prev_freq=prev_freq,
                                                                                               key_padding_mask=key_padding_mask,
                                                                                               attn_mask=attn_mask)
        else:
            output, attn_weights, attn_socres_freqw = self.ITFA(q_s, k_s, v_s, key_padding_mask=key_padding_mask,
                                                                attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1,
                                                          self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores, attn_socres_freqw, attn_socres_freq
        else:
            return output, attn_weights, attn_socres_freqw


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


# TODO 构建时频交互注意力机制
class InteractiveTimeFeqAttention(nn.Module):

    def __init__(self, d_model, n_heads, len_q, len_kv, attn_dropout=0., res_attention=False, lsa=False,
                 activation: Optional[str] = 'tanh'):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa
        self.activation = activation
        self.scale_freq = 1/nn.Parameter(torch.tensor(head_dim ** 2), requires_grad=lsa)
        # 设定频域所需的特征长度
        self.index_q = self.get_freqency_orderlist(len_q, reduction=True)  # 12 -> 6
        self.index_kv = self.get_freqency_orderlist(len_kv, reduction=True)  # 12 -> 6

        # 构建交互尺度变换模块
        self.kv_transform = InteractiveMultiScale([3, 5, 7], len_kv, len_kv, Reduction=None,
                                                  Emdbedding_dim=d_model // n_heads)

        # 用于对频率域的结果进行维度变换
        self.freq_transform = nn.Sequential(
            nn.Linear(len(self.index_q), len_q)
        )
        # fusion模块对三个来源的数据进行合并
        # self.Trifuion = nn.Sequential(
        #     nn.Dropout(0.1),
        #     nn.Linear(len_q, len_q),
        #     nn.LayerNorm(len_q),
        #     nn.ReLU()
        # )
        # 频率域的复数乘法所用权重
        self.learned_matrix1 = self.scale_freq * nn.Parameter(
            torch.rand(n_heads, d_model // n_heads, d_model // n_heads, len(self.index_q),
                       dtype=torch.float))  # size:(n_heads,d_model//n_heads,d_model//n_heads,len(q))
        self.learned_matrix2 = self.scale_freq * nn.Parameter(
            torch.rand(n_heads, d_model // n_heads, d_model // n_heads, len(self.index_q),
                       dtype=torch.float))  # size:(n_heads,d_model//n_heads,d_model//n_heads,len(kv))

    # 随机打乱获得频率域序号
    def get_freqency_orderlist(self, seq_len, init_set=6, reduction: bool = True):
        if reduction:
            assert isinstance(reduction, bool)
            set_len = min(seq_len // 2, init_set)
        else:
            set_len = min(seq_len, init_set)
        range_list = list(range(0, seq_len // 2))
        np.random.shuffle(range_list)
        index = range_list[:set_len]
        index.sort()
        return index

    # 复数乘法，权重构建
    def complex_multiply(self, order, x, weights):
        x_flag = True
        w_flag = True
        if not torch.is_complex(x):
            x_flag = False
            x = torch.complex(x, torch.zeros_like(x).to(x.device))
        if not torch.is_complex(weights):
            w_flag = False
            weights = torch.complex(weights, torch.zeros_like(weights).to(weights.device))
        if x_flag or w_flag:
            return torch.complex(torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag),
                                 torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real))
        else:
            return torch.einsum(order, x.real, weights.real)

    # 傅里叶变换
    def fft(self, input_seq, index_seq):
        # (B,H,S,E) --> (B,H,E,S)
        B, H, S, E = input_seq.shape
        input_seq = input_seq.permute(0, 1, 3, 2)  # 将path_len的长度放到最后进行real傅里叶变换
        seq_x = torch.fft.rfft(input_seq, dim=-1)  # (B,H,E,S) -> (B,H,E,S)
        seq_x_ = torch.zeros((B, H, E, len(index_seq)), dtype=torch.cfloat)
        for index, i in enumerate(index_seq):
            if i >= input_seq.shape[3]:
                continue
            seq_x_[:, :, :, index] = seq_x[:, :, :, i]
        seq_x_ = seq_x_.permute(0, 1, 3, 2)
        return seq_x_

    # 傅里叶反变换计算
    def ifft(self, fft_out):
        fft_out = torch.fft.irfft(fft_out, n=fft_out.shape[3])
        return fft_out

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None,
                prev_freq: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k] :(896,8,12,64)
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # TODO 先对Q,K进行多尺度特征提取
        # B, H, L, E = v.shape
        k = k.permute(0, 1, 3, 2)  # [896,8,64,12] --> [896,8,12,64]
        k = self.kv_transform(k)
        v = self.kv_transform(v)  # [896,8,12,64] -> [896,8,12,64]

        v_hat = v.clone()
        # 获得频率域的q,k,v
        freq_q = self.fft(q, self.index_q)  # [896,8,12,64]
        freq_k = self.fft(k, self.index_kv)
        freq_v = self.fft(v, self.index_kv)  # [896,8,6,64]
        freq_k = freq_k.permute(0, 1, 3, 2)  # [896,8,6,64] --> [896,8,64,6]
        # TODO 计算时间与频率域的attention_score
        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k.permute(0, 1, 3,
                                                2)) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]

        attn_socres_freq = self.complex_multiply("bhse,bhed->bhsd", freq_q, freq_k)
        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev; attn_socres_freq = attn_socres_freq + prev_freq

        # Attention mask (optional)
        if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
                attn_socres_freq.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask
                attn_socres_freq += torch.complex(attn_mask, torch.zeros_like(attn_mask).to(attn_mask.device))

        # Key padding mask (optional)
        if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)
            attn_socres_freq.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        if self.activation == 'tanh':
            attn_socres_freqw = torch.complex(attn_socres_freq.real.tanh(), attn_socres_freq.imag.tanh())
        elif self.activation == 'softmax':
            attn_socres_freqw = torch.softmax(abs(attn_socres_freq), dim=-1)
            attn_socres_freqw = torch.complex(attn_socres_freqw, torch.zeros_like(attn_socres_freq))
        else:
            raise Exception('No such activation for attention_map!')

        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)
        attn_socres_freqw = self.attn_dropout(attn_socres_freqw)
        # TODO 计算频率域输出并进行可学习矩阵的加权
        freq_out = self.complex_multiply("bhqk,bhke->bhqe", attn_socres_freqw, freq_v)  # [b,h,s,e]
        freq_out_weighted = self.complex_multiply("bhqe,heoq->bhoq", freq_out,
                                                  torch.complex(self.learned_matrix1, self.learned_matrix2).to(
                                                      freq_out.device))  # [b,h,e,s]*[h,e,e,s//2]
        B, H, L, E = q.shape
        freq_out_weighted_ = torch.zeros(B, H, E, L // 2, device=q.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            if i >= freq_out_weighted.shape[3] or j >= freq_out_weighted_.shape[3]:
                continue
            freq_out_weighted_[:, :, :, j] = freq_out_weighted[:, :, :, i]
        freq_out = self.ifft(freq_out_weighted_)
        freq_out = self.freq_transform(freq_out)
        freq_out = freq_out.permute(0, 1, 3, 2) * self.scale_freq # [896,8,64,12]->[896,8,12,64]

        # compute the new values given the attention weights
        time_out = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]
        # TODO 最后对v以及时频域的结果进行融合

        # out = self.Trifuion((freq_out + time_out + v_hat).permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out = freq_out + time_out

        if self.res_attention:
            return out, attn_weights, attn_socres_freqw, attn_scores, attn_socres_freq
        else:
            return out, attn_weights, attn_socres_freqw


# 交互多尺度特征提取模块
class InteractiveMultiScale(nn.Module):
    # (B,H,S,E) --> (B*H,S,E)
    def __init__(self, Scales, Input_channel, Output_channel, Reduction, Emdbedding_dim):
        super(InteractiveMultiScale, self).__init__()
        if Reduction is None:
            Reduction = 4
        self.Inter_channel = Input_channel // Reduction
        self.Conv1 = nn.Sequential(
            nn.Conv1d(in_channels=Input_channel, out_channels=self.Inter_channel, kernel_size=1),
            nn.LayerNorm(Emdbedding_dim),  # 对最后一个维度进行归一化
            nn.ReLU()
        )
        self.Scales = Scales
        self.embedding_dim = Emdbedding_dim
        self.padding = [(scale + 1) // 2 for scale in self.Scales]  # 获得使得最后的长度不变的padding
        # 使用多尺度
        self.MultiScale = nn.Sequential(
            *[nn.Conv1d(self.Inter_channel, self.Inter_channel, kernel_size=3, padding=1) for _ in
              range(len(self.Scales))]
        )
        self.Conv_end = nn.Sequential(
            nn.Conv1d(in_channels=self.Inter_channel * len(self.Scales), out_channels=Output_channel, kernel_size=1),
            nn.LayerNorm(Emdbedding_dim),  # 对最后一个维度进行归一化
            nn.ReLU()
        )

    def forward(self, x):
        B, H, S, E = x.shape
        x = x.reshape(-1, S, E)
        init = x.clone()
        x = self.Conv1(x)
        temp = []
        for layer in self.MultiScale:
            x = layer(x)
            temp.append(x)
        x = torch.cat(temp, dim=1)
        x = self.Conv_end(x)
        return (x + init).reshape(B,-1,S,E)







