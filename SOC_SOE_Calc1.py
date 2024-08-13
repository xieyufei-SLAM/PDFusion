import pandas as pd
import numpy as np
import os

def GetRaw(path):
    assert path is not None,'Path is invalid!'
    Context = pd.read_excel(path,sheet_name=3)
    Info = pd.read_excel(path, sheet_name=2)
    ChargeIn_Q,DischargeIn_Q,ChargeIn_E,DischargeIn_E = Info['充电容量(mAh)'],Info['放电容量(mAh)'],\
                                                        Info['充电能量(mWh)'],Info['放电能量(mWh)']
    StanderdQ,StanderdE = np.max(sum(ChargeIn_Q) - sum(DischargeIn_Q)),\
                          np.abs(sum(ChargeIn_E) - sum(DischargeIn_E))
    return StanderdQ,StanderdE,Context

def calc_SOC_SOE(context,StanderdQ,StanderdE):
    valid = context.loc[:, ['绝对时间','相对时间(h:min:s.ms)','状态','容量(mAh)','能量(mWh)']]
    remain = {
        'Q_remain': StanderdQ,
        'E_remain': StanderdE,
    }
    SOCs = []
    SOEs = []
    for index,i in enumerate(valid['状态']):
        if (i == '恒流充电') and (valid['状态'][index] == valid['状态'][index+1]):
            SOC = (valid.loc[index,'容量(mAh)'] + remain['Q_remain']) / StanderdQ
            SOE = (valid.loc[index,'能量(mWh)'] + remain['E_remain']) / StanderdE
        elif (i == '恒流充电') and (valid['状态'][index] != valid['状态'][index+1]): #状态发生转变
            remain['Q_remain'] += valid.loc[index,'容量(mAh)']
            remain['E_remain'] += valid.loc[index,'能量(mWh)']
        elif (i == '恒压充电') and (valid['状态'][index] == valid['状态'][index+1]):
            SOC = (valid.loc[index,'容量(mAh)'] + remain['Q_remain']) / StanderdQ
            SOE = (valid.loc[index,'能量(mWh)'] + remain['E_remain']) / StanderdE
        elif (i == '恒压充电') and (valid['状态'][index] != valid['状态'][index+1]):
            remain['Q_remain'] += valid.loc[index,'容量(mAh)']
            remain['E_remain'] += valid.loc[index,'能量(mWh)']
        elif (i == '恒流放电') and (valid['状态'][index] == valid['状态'][index+1]):
            SOC = (remain['Q_remain'] - valid.loc[index,'容量(mAh)']) / StanderdQ
            SOE = (remain['E_remain'] - valid.loc[index,'能量(mWh)']) / StanderdE
        elif (i == '恒流放电') and (valid['状态'][index] != valid['状态'][index + 1]):
            remain['Q_remain'] -= valid.loc[index,'容量(mAh)']
            remain['E_remain'] -= valid.loc[index,'能量(mWh)']
        elif (i == '恒压放电') and (index != (len(valid['状态'])-1)):
            if (valid['状态'][index] == valid['状态'][index + 1]):
                SOC = (remain['Q_remain'] - valid.loc[index,'容量(mAh)']) / StanderdQ
                SOE = (remain['E_remain'] - valid.loc[index,'能量(mWh)']) / StanderdE
            elif (valid['状态'][index] != valid['状态'][index + 1]):
                remain['Q_remain'] -= valid.loc[index,'容量(mAh)']
                remain['E_remain'] -= valid.loc[index,'能量(mWh)']
        elif (i == '恒压放电') and (index == (len(valid['状态'])-1)):
            SOC = (remain['Q_remain'] - valid.loc[index, '容量(mAh)']) / StanderdQ
            SOE = (remain['E_remain'] - valid.loc[index, '能量(mWh)']) / StanderdE
        else:
            SOC = SOCs[index-1]
            SOE = SOEs[index-1]
        SOCs.append(SOC)
        SOEs.append(SOE)
    return SOCs, SOEs

def resave(context,soc,soe,name):
    soc = pd.DataFrame(soc, columns=['soc'])
    soe = pd.DataFrame(soe, columns=['soe'])
    dataframe = pd.concat([context,soc,soe],axis=1)
    dataframe.to_excel(name)

def main():
    path = 'E:\Datasets\Li_ion\data\WUT_experiments\锂离子电池数据收集excel表格版\hppc实验\hppc'
    save_main = 'E:\Datasets\Li_ion\data\WUT_experiments\锂离子电池数据收集excel表格版\hppc实验\hppc/out'
    for i in os.listdir(path):
        if i.endswith('.xlsx'):
            excel_path = os.path.join(path,i)
            StanderdQ,StanderdE,Context = GetRaw(excel_path)
            SOCs,SOEs = calc_SOC_SOE(Context,StanderdQ,StanderdE)
            save_path = os.path.join(save_main,i)
            resave(Context, SOCs, SOEs, save_path)
            print(f'Excel {i} is processed and saved!')

if __name__ == '__main__':
    main()











