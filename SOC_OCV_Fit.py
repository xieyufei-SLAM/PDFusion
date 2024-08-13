import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def ReadHPPC(path,last_Q,last_E):
    data = pd.read_excel(path,sheet_name='Detail_4_1_7')
    for cycle in data.groupby('循环'):
        cycle_num = cycle[0]
        info = cycle[1]
        info = info.reset_index()
        for j in range(len(info)-1):
            # 计算SOC和SOE值
            if (info.loc[j,'状态'] == '恒流放电'):
                SOC = (last_Q - info.loc[j, '容量(mAh)']) / init_NCM_Q
                SOE = (last_E - info.loc[j, '能量(mWh)']) / init_NCM_E

            elif (info.loc[j,'状态'] == '恒流充电'):
                SOC = (last_Q + info.loc[j, '容量(mAh)']) / init_NCM_Q
                SOE = (last_E + info.loc[j, '能量(mWh)']) / init_NCM_E

            if (info.loc[j,'状态'] == '搁置') and (info.loc[j+1,'状态'] != '搁置'):
                OCV = info.loc[j,'电压(V)']
                OCV_valid.append(OCV)
                SOC_valid.append(SOC)
                SOE_valid.append(SOE)

            if (info.loc[j,'状态'] != info.loc[j+1,'状态']):
                last_Q = SOC * init_NCM_Q
                last_E = SOE * init_NCM_E

    return SOC_valid,SOE_valid,OCV_valid

def curve_reconstruction(x,relations):
    y = np.zeros_like(np.array(x))
    for index,i in enumerate(np.flip(relations)):
        y += i * np.array(x) ** index
    return y

def Fit_SOCSOE_OCV(SOC_valid,SOE_valid,OCV_valid):
    SOC_valid.reverse()
    SOE_valid.reverse()
    OCV_valid.reverse()
    z = np.polyfit(SOC_valid,OCV_valid,7)
    print('SOC多项式系数：',z)
    z1 = np.polyfit(SOE_valid, OCV_valid, 7)
    print('SOE多项式系数：', z1)
    SOX = np.linspace(0, 1, 1000)
    y = curve_reconstruction(SOX, z)
    y1 = curve_reconstruction(SOX, z1)
    plt.title('Voltage-SOC')
    plt.scatter(SOC_valid, OCV_valid, s=20, c='r', linewidth=2, label='SOC-OCV(True)',alpha=0.5)
    plt.plot(SOX, y, '-c', linewidth=2, label='SOC-OCV(fit)')
    savefig = os.path.join('results/ParameterIdentification/','ncm1_SOCOCV.jpg')
    plt.legend(loc='lower right')
    plt.xlabel('SOC')
    plt.ylabel('Voltage/V')
    plt.savefig(savefig, dpi=300)
    plt.show()

    plt.title('Voltage-SOE')
    plt.scatter(SOE_valid, OCV_valid, s=20, c='m', linewidth=2, label='SOE-OCV(True)', alpha=0.5)
    plt.plot(SOX, y1, '-g', linewidth=2, label='SOE-OCV(fit)')
    savefig = os.path.join('results/ParameterIdentification/', 'ncm1_SOEOCV.jpg')
    plt.legend(loc='lower right')
    plt.xlabel('SOE')
    plt.ylabel('Voltage/V')
    plt.savefig(savefig, dpi=300)
    plt.show()

    pass

if __name__ == '__main__':
    # 初值设定
    init_LPF_Q = 1830  # 电池初始容量为1800mAH
    init_LPF_E = 6100  # 电池初始电量为6000mAH

    init_NCM_Q = 2740  # 电池初始容量为1800mAH
    init_NCM_E = 10300  # 电池初始电量为6000mAH

    # 读取HPPC数据
    last_Q = init_NCM_Q
    last_E = init_NCM_E

    SOC_valid = []
    OCV_valid = []
    SOE_valid = []

    path = r'E:\Datasets\Li_ion\data\WUT_experiments\锂离子电池数据收集excel表格版\hppc实验\hppc\1号ncm电池hppc实验（复测）.xlsx'
    SOC_valid,SOE_valid,OCV_valid = ReadHPPC(path,last_Q,last_E)
    hyper_high_low = Fit_SOCSOE_OCV(SOC_valid,SOE_valid,OCV_valid)