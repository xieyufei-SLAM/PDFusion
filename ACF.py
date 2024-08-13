from statsmodels.graphics.tsaplots import plot_acf

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel(r'E:\LSTF_Projects\PatchTST_supervised\data\Li_ion\1号lpf电池0.5，0.5充放电曲线.xlsx',sheet_name=0)
df_raw = pd.concat((df.loc[:,'电流(mA)':'电压(V)'],df.loc[:,'C1':'R2'],df.loc[:,'soc']),axis=1)

temp = pd.concat((df.loc[:,'电流(mA)':'电压(V)'],df.loc[:,'C1':'R2'], df.loc[:,'soc':'soe']),axis=1)
temp.rename(columns={'电流(mA)':'Current'}, inplace = True)
temp.rename(columns={'电压(V)':'Voltage'}, inplace = True)
temp.rename(columns={'soc':'SOC'}, inplace = True)
temp.rename(columns={'soe':'SOE'}, inplace = True)
y = df.loc[:,'soc']

plt.rcParams.update({'figure.dpi': 300})  # 设置图片大小
for i in range(temp.shape[1]):
    plot_acf(temp.iloc[:500,i])  # 生成自相关图
    plt.title(f'{temp.columns[i]}',fontsize=25)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.show()