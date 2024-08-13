import argparse
import os
import torch
from exp.exp_main import Exp_Main
from exp.exp_main_IL import Exp_Main_IL

import random
import numpy as np

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='train', help='model id')

parser.add_argument('--model', type=str, default='PatchTST_TIA',
                    help='model name, options: [iTransformer,ESTformer,TimeMixer, Autoformer, Informer, Transformer, PatchTST_TIA, PatchTST_Base]')
parser.add_argument('--task_name', type=str, default='long_term_forecast', help='task')
parser.add_argument('--down_sampling_layers', type=int, default=2, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default='conv',
                    help='down sampling method, only support avg, max, conv')
parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')

# data loader
parser.add_argument('--data', type=str, default='custom', help='dataset type；custom_NoCorrection, custom，custom_Calce')
parser.add_argument('--root_path', type=str, default=r'data/Li_ion/DCD/lpf/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default=r'1号lpf电池0.5，0.5充放电实验.xlsx', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=list, default=['soc_residual','soe_residual'], help='target feature in S or MS task')

parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

"""
# forecasting task:
1~96时间步为encoder输入 -- seq_len 96
49～96时间步为decoder输入（其中97～192时间步被置为0）-- label_len 48
97～192时间步为decoder输出 -- pred_len 96

其中：
|-------------seq_len:96---------------|
                  |----label_len:48----|
                                       |---------------pred_len:96---------------|
|-----------------|--------------------|-----------------------------------------|
"""
parser.add_argument('--seq_len', type=int, default=32, help='input sequence length')
parser.add_argument('--label_len', type=int, default=16, help='start token length')
parser.add_argument('--pred_len', type=int, default=32, help='prediction sequence length')
# DLinear
#parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=8, help='patch length') # patch长度，设置为16.
parser.add_argument('--stride', type=int, default=8, help='stride') # 步长设置为8
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end') # 在尾部进行padding
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0') # 反归一化操作
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# Formers 
parser.add_argument('--embed_type', type=int, default=1, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=9, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=9, help='decoder input size')
parser.add_argument('--c_out', type=int, default=2, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
parser.add_argument('--if_incremental', default=True, help='if use IL to domain transfer')
parser.add_argument('--if_correction', default=True, help='if use Correction to domain transfer')
parser.add_argument('--source_path', default='1号lpf电池0.5，0.5充放电实验.xlsx', help='Domain_path')
parser.add_argument('--target_path', default='1号lpf电池0.5，1.5充放电实验.xlsx', help='Domain_path')
# 02_26_2016_SP20-2_0C_US06_80SOC.xlsx 11_11_2015_SP20-2_US06_80SOC.xlsx 12_16_2015_SP20-2_45C_US06_80SOC.xlsx
args = parser.parse_args()

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.if_correction is False:
    args.target = ['soc','soe']

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

if __name__ == '__main__':
    print('Args in experiment:')
    print(args)

    # TODO 如果使用增量学习，则对两个域进行对齐
    if args.if_incremental:
        Exp = Exp_Main_IL
    else:
        Exp = Exp_Main
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff, #FFN的维度2048
                args.factor,
                args.embed,
                args.distil,
                args.des,ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # exp.test(setting)
            if args.if_incremental:
                rmse, mae, mape, R2, rmse1, mae1, mape1, R21 = exp.test(setting, test=1, Domain_index='source')
                rmset, maet, mapet, R2t, rmse1t, mae1t, mape1t, R21t = exp.test(setting, test=1, Domain_index='target')
                print(
                    f'\033[92m PDFusion source :average of SOC and SOE\033[0m' + '\033[92m>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\033[0m')
                print(
                    'SOX:   rmse:{}, mae:{} mape: {} R2:{}'.format((rmse + rmse1)*100 / 2,
                                                                   (mae + mae1)*100  / 2,
                                                                   (mape + mape1) / 2,
                                                                   (R2 + R21) / 2))

                print(
                    f'\033[92m PDFusion target :average of SOC and SOE\033[0m' + '\033[92m>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\033[0m')
                print(
                    'SOX:   rmse:{}, mae:{} mape: {} R2:{}'.format((rmset + rmse1t)*100  / 2,
                                                                   (maet + mae1t)*100  / 2,
                                                                   (mapet + mape1t) / 2,
                                                                   (R2t + R21t) / 2))
            else:
                rmse, mae, mape, R2, rmse1, mae1, mape1, R21 = exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                      args.model,
                                                                                                      args.data,
                                                                                                      args.features,
                                                                                                      args.seq_len,
                                                                                                      args.label_len,
                                                                                                      args.pred_len,
                                                                                                      args.d_model,
                                                                                                      args.n_heads,
                                                                                                      args.e_layers,
                                                                                                      args.d_layers,
                                                                                                      args.d_ff,
                                                                                                      args.factor,
                                                                                                      args.embed,
                                                                                                      args.distil,
                                                                                                      args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        if args.if_incremental:
            rmse, mae, mape, R2, rmse1, mae1, mape1, R21 = exp.test(setting, test=1, Domain_index='source')
            rmset, maet, mapet, R2t, rmse1t, mae1t, mape1t, R21t = exp.test(setting, test=1, Domain_index='target')
            print(
                f'\033[92m PDFusion source :average of SOC and SOE\033[0m' + '\033[92m>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\033[0m')
            print(
                'SOX:   rmse:{}, mae:{} mape: {} R2:{}'.format((rmse + rmse1) / 2,
                                                               (mae + mae1) / 2,
                                                               (mape + mape1) / 2,
                                                               (R2 + R21) / 2))

            print(
                f'\033[92m PDFusion target :average of SOC and SOE\033[0m' + '\033[92m>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\033[0m')
            print(
                'SOX:   rmse:{}, mae:{} mape: {} R2:{}'.format((rmset + rmse1t) / 2,
                                                               (maet + mae1t) / 2,
                                                               (mapet + mape1t) / 2,
                                                               (R2t + R21t) / 2))
        else:
            rmse, mae, mape, R2, rmse1, mae1, mape1, R21 = exp.test(setting)
        torch.cuda.empty_cache()


