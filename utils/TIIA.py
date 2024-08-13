import torch
from torch import nn

"""
已经在模型中集成，此处只作为参考，没有跳转和继承
"""
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
        self.index_q = len_q//4
        self.index_kv = len_q//4

        # 构建交互尺度变换模块
        self.kv_transform = InteractiveMultiScale([3, 5, 7], len_kv, len_kv, Reduction=None,
                                                  Emdbedding_dim=d_model // n_heads)

        # 用于对频率域的结果进行维度变换
        self.freq_transform = nn.Sequential(
            nn.Linear(self.index_q, len_q)
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
            torch.rand(n_heads, d_model // n_heads, d_model // n_heads, self.index_q,
                       dtype=torch.float))  # size:(n_heads,d_model//n_heads,d_model//n_heads,len(q))
        self.learned_matrix2 = self.scale_freq * nn.Parameter(
            torch.rand(n_heads, d_model // n_heads, d_model // n_heads, self.index_q,
                       dtype=torch.float))  # size:(n_heads,d_model//n_heads,d_model//n_heads,len(kv))

    # TODO 改进点2：频域FFT变换与重采样 -> 使用topk获得有用的序列编码
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

    # TODO 仅使用高频信息
    def rfft_topk(self, in_seq, seq_len, topk=None):
        B,H,L,E = in_seq.shape
        if topk is None:
            topk = seq_len // 2
        in_seq = in_seq.permute(0, 1, 3, 2) # [B,H,E,L]
        seq_x = torch.fft.rfft(in_seq,dim=-1)
        frequency_list = abs(seq_x).mean(0).mean(0).mean(0)
        frequency_list[0] = 0
        seqlist,topk = torch.topk(frequency_list,k=topk,largest=True)
        topk = topk.cpu().numpy()
        seq_x_ = torch.zeros((B, H, E, len(topk)), dtype=torch.cfloat)
        for index, i in enumerate(topk):
            if i >= in_seq.shape[3]:
                continue
            seq_x_[:, :, :, index] = seq_x[:, :, :, i]
        return seq_x_.permute(0,1,3,2)

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
        B, H, L, E = v.shape
        k = k.permute(0, 1, 3, 2)  # [896,8,64,12] --> [896,8,12,64]
        k = self.kv_transform(k)
        v = self.kv_transform(v)  # [896,8,12,64] -> [896,8,12,64]
        # 获得频率域的q,k,v
        tk_len = L//4
        freq_q = self.rfft_topk(q,seq_len=L, topk=tk_len)
        freq_k = self.rfft_topk(q, seq_len=L, topk=tk_len)
        freq_v = self.rfft_topk(q, seq_len=L, topk=tk_len)
        # freq_q = self.fft(q, self.index_q)  # [896,8,12,64]
        # freq_k = self.fft(k, self.index_kv)
        # freq_v = self.fft(v, self.index_kv)  # [896,8,6,64]
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
        freq_out_weighted_ = torch.zeros(B, H, E, tk_len, device=q.device, dtype=torch.cfloat)
        for i, j in enumerate(range(self.index_q)):
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