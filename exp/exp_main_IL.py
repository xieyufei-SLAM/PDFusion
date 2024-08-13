from data_provider.data_factory import data_provider
from exp.exp_basic_IL import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, Li_ionFormer, PatchTST_Baseline, itransformer,ESTformer,TimeMIxer
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from layers.Domain_Adaptor import Domain_Adaptor
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader, Dataset


class BalancedDataLoader:
    def __init__(self, dataset1_loader, dataset2_loader):
        self.dataset1_loader = dataset1_loader
        self.dataset2_loader = dataset2_loader
        self.iterator1 = iter(dataset1_loader)
        self.iterator2 = iter(dataset2_loader)

    def __iter__(self):
        self.iterator1 = iter(self.dataset1_loader)
        self.iterator2 = iter(self.dataset2_loader)
        return self

    def __next__(self):
        try:
            data1 = next(self.iterator1)
        except StopIteration:
            raise StopIteration

        try:
            data2 = next(self.iterator2)
        except StopIteration:
            raise StopIteration

        return data1, data2

    def __len__(self):
        return min(len(self.dataset1_loader), len(self.dataset2_loader))



class Exp_Main_IL(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main_IL, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST_Base': PatchTST_Baseline,
            'PatchTST_TIA': Li_ionFormer,
            'iTransformer': itransformer,
            'ESTformer':ESTformer,
            'TimeMixer':TimeMIxer
        }
        model = model_dict[self.args.model].Model(self.args).float()
        self.DomainAdaptor = Domain_Adaptor(5,16,9,nvar=9)
        self.DomainAdaptor = self.DomainAdaptor.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model,self.DomainAdaptor

    def _get_data(self, flag, Domain_index):
        data_set, data_loader = data_provider(self.args, flag, Domain_index)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device) # [b,label_len + pred_len,d_model]
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs,inners = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs,inners = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)


                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train',Domain_index='source')
        vali_data, vali_loader = self._get_data(flag='val',Domain_index='source')
        test_data, test_loader = self._get_data(flag='test',Domain_index='source')

        train_data_target, train_loader_target = self._get_data(flag='train',Domain_index='target')
        vali_data_target, vali_loader_target = self._get_data(flag='val',Domain_index='target')
        test_data_target, test_loader_target = self._get_data(flag='test',Domain_index='target')

        BD = BalancedDataLoader(train_loader,train_loader_target)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            for i, (source_batch, target_batch) in enumerate(BD):
                batch_x, batch_y, batch_x_mark, batch_y_mark = source_batch
                batch_x_tar, batch_y_tar, batch_x_mark_tar, batch_y_mark_tar = target_batch
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x_tar = batch_x_tar.float().to(self.device)
                batch_y_tar = batch_y_tar.float().to(self.device)
                batch_x_mark_tar = batch_x_mark_tar.float().to(self.device)
                batch_y_mark_tar = batch_y_mark_tar.float().to(self.device)

                # decoder input
                """
                其他的former采用的是将label_len和pred_len拼起来作为decoder的输入
                所以这里的dec_inp:[128, 144, 7],并且后面的pred_len初始化时用的是全0
                """
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs,inners = self.model(batch_x)
                            outputs_tar, inners_tar = self.model(batch_x_tar)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                outputs_tar = self.model(batch_x_tar, batch_y_tar, batch_x_mark_tar, batch_y_mark_tar)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                outputs_tar = self.model(batch_x_tar, batch_y_tar, batch_x_mark_tar, batch_y_mark_tar)

                        f_dim = -1 if self.args.features == 'MS' else -2
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]  #[128.96,7]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  #[128.96,7]

                        outputs_tar = outputs_tar[:, -self.args.pred_len:, f_dim:]  # [128.96,7]
                        batch_y_tar = batch_y_tar[:, -self.args.pred_len:, f_dim:].to(self.device)  # [128.96,7]

                        loss = criterion(outputs, batch_y) + criterion(outputs_tar, batch_y_tar) + self.DomainAdaptor.forward(outputs,inners,outputs_tar,inners_tar)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs, inners = self.model(batch_x)
                        outputs_tar, inners_tar = self.model(batch_x_tar)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else -2

                    outputs = outputs[:, -self.args.pred_len:, f_dim:]  #[128.96,7]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  #[128.96,7]

                    outputs_tar = outputs_tar[:, -self.args.pred_len:, f_dim:]  # [128.96,7]
                    batch_y_tar = batch_y_tar[:, -self.args.pred_len:, f_dim:].to(self.device)  # [128.96,7]

                    loss = criterion(outputs, batch_y) + criterion(outputs_tar, batch_y_tar) + 1e-5 * self.DomainAdaptor.forward(outputs, inners, outputs_tar, inners_tar)

                    # loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward(retain_graph=True)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward(retain_graph=True)
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            vali_loss_target = self.vali(vali_data_target, vali_loader_target, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            test_loss_target = self.vali(test_data_target, test_loader_target, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss+vali_loss_target, test_loss+test_loss_target))
            early_stopping(vali_loss + vali_loss_target, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0, Domain_index='source'):
        test_data, test_loader = self._get_data(flag='test',Domain_index=Domain_index)
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        gt = []
        pd=[]
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs,inners = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs,inners = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else -2
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # outputs = outputs[:, :, f_dim:]
                # batch_y = batch_y[:, :, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

                # TODO 直接输出测试集下的输出
                input = batch_x.detach().cpu().numpy()
                gt.append(true[:, -1, f_dim:])
                pd.append(pred[:, -1, f_dim:])
            gt = np.vstack(gt)
            pd = np.vstack(pd)
            gt_pad = np.zeros((gt.shape[0], test_data.data_x.shape[1]))
            gt_pad[:, -2:] = gt
            pd_pad = np.zeros((pd.shape[0], test_data.data_x.shape[1]))
            pd_pad[:, -2:] = pd

            gt = test_data.inverse_transform(gt_pad)[:, -2:]
            pd = test_data.inverse_transform(pd_pad)[:, -2:]
            # gt = np.concatenate((gt, true[0, :-1, -1]), axis=0)
            # pd = np.concatenate((pd, pred[0, :-1, -1]), axis=0)
            data_path = os.path.join(self.args.root_path, self.args.data_path)
            save_path = 'test_results/new_collection_IL'
            rmse, mae, mape, R2, rmse1, mae1, mape1, R21 = visual(gt, pd, os.path.join(folder_path, str('test') + '.pdf'), data_path=data_path,
                   seq_len=self.args.seq_len, save_path=save_path, excel_name=self.args.data_path)
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae2, mse2, rmse2, mape2, mspe2, rse2, corr2, R22 = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{} rmse:{} R2:{}'.format(mse2, mae2, rse2, rmse2, R22))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{} rmse:{} R2:{}'.format(mse2, mae2, rse2, rmse2, R22))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return rmse, mae, mape, R2, rmse1, mae1, mape1, R21

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred',Domain_index='source')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs,inners = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs,inners = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
