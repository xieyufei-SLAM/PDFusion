import os.path

import torch.nn as nn
import torch

#TODO 使用域自适应模块对源域和目标域的值进行对齐
"""
step1.使用域自适应模块对源域和目标域的值进行对齐
step2.计算最大均值差异（MMD）以计算源域和目标域在可重现的核希尔伯特空间（RKHS）中的分布差异
"""
# 对注意力图和结果都进行对齐
import torch
import torch.nn.functional as F


class MMDLoss(nn.Module):
    def __init__(self, sigma=1.0):
        """
        初始化MMD损失计算模块。

        参数:
        sigma: float, 高斯核的带宽
        """
        super(MMDLoss, self).__init__()
        self.sigma = sigma

    def hilbert_kernel(self, x, y):
        """
        计算希尔伯特空间映射后的核矩阵。

        参数:
        x: tensor, 输入数据 (b, d)
        y: tensor, 输入数据 (b, d)

        返回:
        kernel_matrix: tensor, 希尔伯特核空间映射后的核矩阵 (b, b)
        """
        # 计算 x 和 y 的内积
        x_norm = torch.sum(x ** 2, dim=1).view(-1, 1)  # (b, 1)
        y_norm = torch.sum(y ** 2, dim=1).view(1, -1)  # (1, b)
        xy = torch.matmul(x, y.T)  # (b, b)

        # 计算距离矩阵的平方
        dist_sq = x_norm + y_norm - 2 * xy

        # 计算高斯核矩阵
        kernel_matrix = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        return kernel_matrix

    def forward(self, source, target):
        """
        计算源域和目标域样本之间的最大均值差异（MMD）。

        参数:
        source: tensor, 源域样本 (b, s, e)
        target: tensor, 目标域样本 (b, s, e)

        返回:
        mmd: tensor, MMD值
        """
        b, s, e = source.size()

        # 将输入数据调整为二维
        source = source.reshape(b, s * e)
        target = target.reshape(b, s * e)

        # 计算希尔伯特核空间映射后的核矩阵
        kernel_ss = self.hilbert_kernel(source, source)
        kernel_tt = self.hilbert_kernel(target, target)
        kernel_st = self.hilbert_kernel(source, target)

        # 计算MMD
        mmd = torch.mean(kernel_ss) + torch.mean(kernel_tt) - 2 * torch.mean(kernel_st)
        mmd = torch.sqrt(torch.abs(mmd))

        return mmd


MMD = MMDLoss()

def adaptive_padd(kernelsize):
    return (kernelsize - 1) // 2

class Domain_Adaptor(nn.Module):
    def __init__(self,cin,cout,cin1,kernel=3,embed_dim=512,nvar=9):
        super(Domain_Adaptor, self).__init__()
        self.nvar = nvar
        self.Encoder = nn.MultiheadAttention(embed_dim=embed_dim,num_heads=8,dropout=0.1,batch_first=True)

        self.Attention_mapper = nn.Sequential(
            nn.Conv1d(in_channels=cin,out_channels=cout,kernel_size=kernel,padding=adaptive_padd(kernel)),
            nn.ReLU()
        )
        self.Attention_Freq_mapper = nn.Sequential(
            nn.Conv1d(in_channels=cin1, out_channels=cout, kernel_size=kernel, padding=adaptive_padd(kernel)),
            nn.ReLU()
        )

        for m in self.Attention_mapper:
            if isinstance(m,nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight,mode='fan_in',nonlinearity='leaky_relu'
                )
        for m in self.Attention_Freq_mapper:
            if isinstance(m,nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight,mode='fan_in',nonlinearity='leaky_relu'
                )

    def forward(self,Out, Inners, Out1, Inners2):
        bv, pnum, plen, plen = Inners['score'].shape
        bv, pnum, _, _  = Inners['freq_score'].shape
        Inners['score'] = Inners['score'].reshape(bv*pnum, plen, -1)
        Inners2['score'] = Inners2['score'].reshape(bv * pnum, plen, -1)
        Inners['freq_score'] = Inners['freq_score'].reshape(bv // self.nvar, self.nvar, -1)
        Inners2['freq_score'] = Inners2['freq_score'].reshape(bv // self.nvar, self.nvar, -1)

        show_freqhotmap = torch.abs(Inners['freq_score'])
        show_timehotmap = Inners['score'].reshape(bv, pnum, plen, -1).reshape(bv // self.nvar,self.nvar,-1)

        # 归一化函数
        def normalize(tensor):
            min_val = tensor.min()
            max_val = tensor.max()
            return (tensor - min_val) / (max_val - min_val)

        # 归一化数据
        show_timehotmap = normalize(show_timehotmap)
        show_freqhotmap = normalize(show_freqhotmap)
        # 绘制热力图
        import matplotlib.pyplot as plt
        import seaborn as sns
        # plt.figure(figsize=(12, 6))

        labels = ['Current', 'Voltage', 'C1', 'C2', 'R0', 'R1', 'R2', 'SOC(history)', 'SOE(history)']

        # 设置字体
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 15

        save_main_t = 'heatmap/time'
        save_main_f = 'heatmap/freq'
        for i in range(show_freqhotmap.shape[0]):
            # 绘制时域热力图
            plt.figure()
            ax1 = sns.heatmap(show_timehotmap[i, :, :show_freqhotmap.shape[-1]].detach().cpu().numpy(),
                        cmap='viridis', yticklabels=labels)
            ax1.set_xticks([])  # 隐藏横轴坐标
            ax1.set_xlabel('')  # 移除横轴标签
            plt.tight_layout()
            # plt.title('Time Domain Heatmap')
            save_t = os.path.join(save_main_t,f'{i}_time.jpg')
            plt.savefig(save_t, dpi=300)

            # 绘制频域热力图
            plt.figure()
            ax2 = sns.heatmap(show_freqhotmap[i].detach().cpu().numpy(), cmap='viridis', yticklabels=labels)
            ax2.set_xticks([])  # 隐藏横轴坐标
            ax2.set_xlabel('')  # 移除横轴标签
            # plt.title('Frequency Domain Heatmap')
            save_f = os.path.join(save_main_f,f'{i}_freq.jpg')
            plt.tight_layout()
            plt.savefig(save_f, dpi=300)

        self.at = self.Attention_mapper(Inners['score'])
        self.at2 = self.Attention_mapper(Inners2['score'])

        self.at_freq = self.Attention_Freq_mapper(torch.abs(Inners['freq_score']))
        self.at2_freq = self.Attention_Freq_mapper(torch.abs(Inners2['freq_score']))

        self.out_en, _ = self.Encoder(Inners['inners_out'],Inners2['inners_out'],Inners2['inners_out'])
        self.out_en_2, _ = self.Encoder(Inners2['inners_out'], Inners['inners_out'], Inners['inners_out'])


        return MMD(self.out_en,self.out_en_2) + MMD(self.at,self.at2) + MMD(self.at_freq,self.at2_freq)




